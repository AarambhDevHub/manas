//! Fixed Stage 3 training fixtures promoted from the standalone proof.

pub const EMBED_DIM: usize = 32;
pub const HIDDEN_DIM: usize = 64;
pub const OUTPUT_DIM: usize = 32;
pub const LEARNING_RATE: f32 = 0.01;
pub const ANCHOR_TRAIN_EPOCHS: usize = 300;
pub const NOISE_TRAIN_EPOCHS: usize = 200;
pub const ANCHOR_SURVIVAL_THRESHOLD: f32 = 0.65;
pub const NEW_FACT_THRESHOLD: f32 = 0.70;
pub const MAX_FORGETTING_DELTA: f32 = 0.15;
pub const ANCHOR_NEURONS_PER_FACT: usize = 1;
pub const SEEDS: [u64; 5] = [1, 7, 42, 2026, 99_991];

pub const ANCHOR_FACTS: [(&str, &str); 5] = [
    ("cat", "small animal with fur"),
    ("paris", "city in france"),
    ("rust", "systems programming language"),
    ("everest", "highest mountain on earth"),
    ("dna", "double helix genetic information"),
];

pub const NOISE_FACTS: [(&str, &str); 50] = [
    ("amazon", "largest river by discharge"),
    ("einstein", "developed theory of relativity"),
    ("photosynthesis", "converts sunlight to energy"),
    ("hydrogen", "lightest element in universe"),
    ("brain", "contains billions of neurons"),
    ("shakespeare", "wrote plays and sonnets"),
    ("light", "travels at constant speed"),
    ("rome", "empire fell in ancient history"),
    ("water", "boils at standard pressure"),
    ("python", "created by guido van rossum"),
    ("jupiter", "largest planet solar system"),
    ("mona lisa", "painted by leonardo da vinci"),
    ("mitochondria", "powerhouse of cell"),
    ("pacific", "largest ocean on earth"),
    ("bitcoin", "created by satoshi nakamoto"),
    ("nitrogen", "moves through ecosystems"),
    ("gravity", "pulls objects toward mass"),
    ("moon", "orbits planet earth"),
    ("mars", "red planet with thin atmosphere"),
    ("venus", "hot planet with dense atmosphere"),
    ("saturn", "planet with visible rings"),
    ("mercury", "closest planet to sun"),
    ("oxygen", "gas required for respiration"),
    ("carbon", "basis of organic chemistry"),
    ("helium", "noble gas used in balloons"),
    ("sodium", "reactive metal in salt"),
    ("chlorine", "greenish gas disinfects water"),
    ("glucose", "sugar used for energy"),
    ("protein", "molecule made of amino acids"),
    ("cell", "basic unit of life"),
    ("bacteria", "single celled microorganisms"),
    ("virus", "infectious particle needing host"),
    ("volcano", "erupts molten rock"),
    ("earthquake", "shaking from tectonic movement"),
    ("hurricane", "rotating tropical storm"),
    ("desert", "dry region with little rainfall"),
    ("rainforest", "dense forest with high rainfall"),
    ("tundra", "cold biome with permafrost"),
    ("democracy", "government by elected people"),
    ("currency", "medium used for exchange"),
    ("algorithm", "step by step procedure"),
    ("compiler", "translates source code"),
    ("database", "stores structured information"),
    ("network", "connects computers together"),
    ("encryption", "protects data with keys"),
    ("battery", "stores electrical energy"),
    ("magnet", "produces magnetic field"),
    ("telescope", "observes distant objects"),
    ("microscope", "magnifies tiny objects"),
    ("thermometer", "measures temperature"),
];
