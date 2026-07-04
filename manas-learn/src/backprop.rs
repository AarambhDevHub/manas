use manas_core::{ManasError, Network, NeuronGradients};

/// Compute mean squared error between two vectors.
pub fn mse_loss(actual: &[f32], expected: &[f32]) -> Result<f32, ManasError> {
    if actual.len() != expected.len() {
        return Err(ManasError::InvalidNetwork(format!(
            "loss vector mismatch: actual {}, expected {}",
            actual.len(),
            expected.len()
        )));
    }

    Ok(actual
        .iter()
        .zip(expected.iter())
        .map(|(left, right)| {
            let error = left - right;
            error * error
        })
        .sum::<f32>()
        / actual.len() as f32)
}

/// Compute gradients for the associative network.
pub fn compute_gradients(
    network: &Network,
    input: &[f32],
    target: &[f32],
) -> Result<(f32, Vec<(u64, NeuronGradients)>), ManasError> {
    if network.layers.len() < 2 {
        return Err(ManasError::InvalidNetwork(
            "trainer expects at least one hidden layer and one output layer".to_string(),
        ));
    }
    if target.len() != network.output_dim {
        return Err(ManasError::InvalidNetwork(format!(
            "target dimension mismatch: expected {}, found {}",
            network.output_dim,
            target.len()
        )));
    }

    let output_layer_index = network.layers.len() - 1;
    let output_layer = &network.layers[output_layer_index];
    if output_layer.neurons.len() != network.output_dim {
        return Err(ManasError::InvalidNetwork(format!(
            "output layer has {} neurons, expected {}",
            output_layer.neurons.len(),
            network.output_dim
        )));
    }

    let cache = network.forward_with_cache(input);
    let loss = mse_loss(&cache.output, target)?;

    let output_deltas = cache
        .output
        .iter()
        .zip(target.iter())
        .zip(output_layer.neurons.iter())
        .map(|((actual, expected), neuron)| {
            let error_grad = 2.0 * (actual - expected) / network.output_dim as f32;
            error_grad * neuron.derivative_from_output(*actual)
        })
        .collect::<Vec<_>>();

    let mut hidden_deltas = cache
        .hidden_layers
        .iter()
        .map(|layer| vec![0.0; layer.len()])
        .collect::<Vec<_>>();
    for output_neuron in &output_layer.neurons {
        if output_neuron.weights.len() != cache.hidden.len() {
            return Err(ManasError::InvalidNetwork(format!(
                "output neuron {} has {} weights, expected {}",
                output_neuron.id,
                output_neuron.weights.len(),
                cache.hidden.len()
            )));
        }
    }

    let mut global_hidden_index = 0;
    for layer_deltas in &mut hidden_deltas {
        for hidden_delta in layer_deltas {
            for (output_delta, output_neuron) in
                output_deltas.iter().zip(output_layer.neurons.iter())
            {
                *hidden_delta += output_delta * output_neuron.weights[global_hidden_index];
            }
            global_hidden_index += 1;
        }
    }

    for layer_index in (0..output_layer_index).rev() {
        for (neuron_index, hidden_delta) in hidden_deltas[layer_index].iter_mut().enumerate() {
            let activation = cache.hidden_layers[layer_index][neuron_index];
            let neuron = &network.layers[layer_index].neurons[neuron_index];
            *hidden_delta *= neuron.derivative_from_output(activation);
        }

        if layer_index > 0 {
            let current_deltas = hidden_deltas[layer_index].clone();
            for (neuron, delta) in network.layers[layer_index]
                .neurons
                .iter()
                .zip(current_deltas.iter())
            {
                for (previous_delta, weight) in hidden_deltas[layer_index - 1]
                    .iter_mut()
                    .zip(neuron.weights.iter())
                {
                    *previous_delta += delta * weight;
                }
            }
        }
    }

    let hidden_neuron_count = network.layers[..output_layer_index]
        .iter()
        .map(|layer| layer.neurons.len())
        .sum::<usize>();
    let mut gradients = Vec::with_capacity(hidden_neuron_count + output_layer.neurons.len());

    for (output_neuron, output_delta) in output_layer.neurons.iter().zip(output_deltas.iter()) {
        gradients.push((
            output_neuron.id,
            NeuronGradients {
                weight_gradients: cache
                    .hidden
                    .iter()
                    .map(|hidden_value| output_delta * hidden_value)
                    .collect(),
                bias_gradient: *output_delta,
            },
        ));
    }

    for (layer_index, layer_deltas) in hidden_deltas.iter().enumerate().take(output_layer_index) {
        let layer_input = if layer_index == 0 {
            &cache.input
        } else {
            &cache.hidden_layers[layer_index - 1]
        };

        for (hidden_neuron, hidden_delta) in network.layers[layer_index]
            .neurons
            .iter()
            .zip(layer_deltas.iter())
        {
            gradients.push((
                hidden_neuron.id,
                NeuronGradients {
                    weight_gradients: layer_input
                        .iter()
                        .map(|input_value| hidden_delta * input_value)
                        .collect(),
                    bias_gradient: *hidden_delta,
                },
            ));
        }
    }

    Ok((loss, gradients))
}

pub fn cosine(left: &[f32], right: &[f32]) -> f32 {
    let numerator = dot(left, right);
    let left_norm = dot(left, left).sqrt();
    let right_norm = dot(right, right).sqrt();

    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        numerator / (left_norm * right_norm)
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use manas_core::Network;

    #[test]
    fn layer_growth_gradients_support_deep_network() {
        let mut network = Network::new_empty(4);
        network.grow_neuron(0, 4).unwrap();
        network.grow_layer(1, 1).unwrap();

        let input = [0.5, -0.25, 0.75, 0.1];
        let target = [0.1, 0.2, 0.3, 0.4];
        let (loss, gradients) = compute_gradients(&network, &input, &target).unwrap();

        assert!(loss.is_finite());
        assert_eq!(gradients.len(), network.neuron_count() as usize);
        assert!(gradients.iter().all(|(_, gradient)| {
            gradient
                .weight_gradients
                .iter()
                .all(|value| value.is_finite())
                && gradient.bias_gradient.is_finite()
        }));
    }
}
