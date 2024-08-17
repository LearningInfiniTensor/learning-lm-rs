use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{Dtype, SafeTensors};
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    //usage
    //let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
    // let safetensor = SafeTensors::deserialize(&model_file).unwrap();
    // let params = LLamaParams::from_safetensors(&safetensor, &config);
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor: Box<dyn Fn(&str) -> Tensor<f32>> = Box::new(|name: &str| {
            let cur_tensor = safetensor.tensor(name).unwrap();
            assert!(cur_tensor.dtype() == Dtype::F32); //你需要以FP32的形式读取出来
            // 确保字节数量是 f32 的倍数
            let byte_slice = cur_tensor.data().to_vec();
            assert!(byte_slice.len() % 4 == 0);
            // 将字节切片转换为 f32 向量
            let float_vec: Vec<f32> = byte_slice
                .chunks(4) // 每四个字节划分为一组
                .map(|chunk| {
                    let bytes: [u8; 4] = chunk.try_into().expect("Invalid chunk size");
                    f32::from_ne_bytes(bytes) // 将字节转换为 f32
                })
            .collect();
            Tensor::new(float_vec, &cur_tensor.shape().to_vec())
        });

        //Vec<Tensor<T>>
        //以rms att w为例，它有两层，分别是model.layers.0.input_layernorm.weight和model.layers.1.input_layernorm.weight
        //到底有多少层，由config.json控制
        let get_tensor_vec: Box<dyn Fn(&str, usize) -> Vec<Tensor<f32>>> = Box::new(|name: &str, size: usize| {
            (0..size)
                .map(|i| {
                    let tensor_name = format!("model.layers.{i}.{}", name);
                    get_tensor(&tensor_name)
                })
                .collect()
        });


        println!("{}",config.num_hidden_layers);

        //tensor or vector of tensor
        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: get_tensor_vec("input_layernorm.weight",config.num_hidden_layers),
            wq: get_tensor_vec("self_attn.q_proj.weight",config.num_hidden_layers),
            wk: get_tensor_vec("self_attn.k_proj.weight",config.num_hidden_layers),
            wv: get_tensor_vec("self_attn.v_proj.weight",config.num_hidden_layers),
            wo: get_tensor_vec("self_attn.o_proj.weight",config.num_hidden_layers),
            w_up: get_tensor_vec("mlp.up_proj.weight",config.num_hidden_layers),
            w_down: get_tensor_vec("mlp.down_proj.weight",config.num_hidden_layers),
            w_gate: get_tensor_vec("mlp.gate_proj.weight",config.num_hidden_layers),
            rms_ffn_w: get_tensor_vec("post_attention_layernorm.weight",config.num_hidden_layers),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight")
        }
    }
}