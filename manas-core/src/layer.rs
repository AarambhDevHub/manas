use crate::activation::Activation;
use crate::neuron::Neuron;

/// A network layer containing neurons with a default activation label.
#[derive(Clone, Debug)]
pub struct Layer {
    pub id: u32,
    pub neurons: Vec<Neuron>,
    pub activation: Activation,
}

impl Layer {
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.activate(input))
            .collect()
    }
}
