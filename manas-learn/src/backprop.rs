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

/// Compute gradients for the proven two-layer associative network.
pub fn compute_gradients(
    network: &Network,
    input: &[f32],
    target: &[f32],
) -> Result<(f32, Vec<(u64, NeuronGradients)>), ManasError> {
    if network.layers.len() != 2 {
        return Err(ManasError::InvalidNetwork(
            "Stage 3 trainer expects a two-layer network".to_string(),
        ));
    }
    if target.len() != network.output_dim {
        return Err(ManasError::InvalidNetwork(format!(
            "target dimension mismatch: expected {}, found {}",
            network.output_dim,
            target.len()
        )));
    }

    let cache = network.forward_with_cache(input);
    let loss = mse_loss(&cache.output, target)?;

    let output_deltas = cache
        .output
        .iter()
        .zip(target.iter())
        .zip(network.layers[1].neurons.iter())
        .map(|((actual, expected), neuron)| {
            let error_grad = 2.0 * (actual - expected) / network.output_dim as f32;
            error_grad * neuron.derivative_from_output(*actual)
        })
        .collect::<Vec<_>>();

    let mut hidden_deltas = vec![0.0; network.hidden_dim];
    for (output_delta, output_neuron) in output_deltas.iter().zip(network.layers[1].neurons.iter())
    {
        for (hidden_delta, weight) in hidden_deltas.iter_mut().zip(output_neuron.weights.iter()) {
            *hidden_delta += output_delta * weight;
        }
    }

    for ((hidden_delta, hidden_activation), hidden_neuron) in hidden_deltas
        .iter_mut()
        .zip(cache.hidden.iter())
        .zip(network.layers[0].neurons.iter())
    {
        *hidden_delta *= hidden_neuron.derivative_from_output(*hidden_activation);
    }

    let mut gradients =
        Vec::with_capacity(network.layers[0].neurons.len() + network.layers[1].neurons.len());

    for (output_neuron, output_delta) in network.layers[1].neurons.iter().zip(output_deltas.iter())
    {
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

    for (hidden_neuron, hidden_delta) in network.layers[0].neurons.iter().zip(hidden_deltas.iter())
    {
        gradients.push((
            hidden_neuron.id,
            NeuronGradients {
                weight_gradients: cache
                    .input
                    .iter()
                    .map(|input_value| hidden_delta * input_value)
                    .collect(),
                bias_gradient: *hidden_delta,
            },
        ));
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
