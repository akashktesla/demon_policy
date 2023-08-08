#![allow(warnings)]
use tch::Tensor;
use tch::nn;
use tch::nn::{Sequential,Module,VarStore};
use tch::Device::Cpu;

pub fn main(){
    let vs = VarStore::new(Cpu);
    let layers = vec![];
    let p1 = Policy::new(vs,layers);


}

#[derive(Debug)]
struct Policy{
    vs:VarStore,
    network:Sequential,
    layers:Vec<i32>,
}

impl Policy{
    fn new(vs:VarStore,layers:Vec<i32>)->Self{
        return Policy {
            vs,
            network:nn::seq(),
            layers:Vec::new(),
        };
    }

    fn build(&mut self,vs:VarStore){
        let network = nn::seq();
        for i in &self.layers{

        }
    }
}

impl Module for Policy{
    fn forward(&self, xs: &Tensor) -> Tensor {
        return self.network.forward(xs);
    }
}
