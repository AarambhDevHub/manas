/// Activation functions supported by the core network.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
    Keyed,
}

impl Activation {
    /// Apply the activation to a scalar pre-activation value.
    pub fn apply(self, value: f32) -> f32 {
        match self {
            Self::ReLU => value.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-value).exp()),
            Self::Tanh => value.tanh(),
            Self::Linear | Self::Keyed => value,
        }
    }

    /// Compute derivative from the already-computed activation output.
    pub fn derivative_from_output(self, output: f32) -> f32 {
        match self {
            Self::ReLU => {
                if output > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Sigmoid => output * (1.0 - output),
            Self::Tanh => 1.0 - output * output,
            Self::Linear => 1.0,
            Self::Keyed => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tanh_derivative_uses_activation_output() {
        let out = Activation::Tanh.apply(0.5);
        let derivative = Activation::Tanh.derivative_from_output(out);
        assert!(derivative > 0.0);
        assert!(derivative < 1.0);
    }

    #[test]
    fn linear_derivative_is_one() {
        assert_eq!(Activation::Linear.derivative_from_output(42.0), 1.0);
    }
}
