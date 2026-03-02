use candle_core::Device;
use candle_core::Tensor;
use std::error::Error;
use std::fmt::format;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::string;

/*
Gaurav Sablok
codeprog@icloud.com
*/

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct DeviceTensor {
    pub pathfile: String,
}

impl DeviceTensor {
    pub fn tensorcreate(&self) -> Result<Tensor<f64>, Box<dyn Error>> {
        let fileopen = DeviceTensor { pathfile };
        let fileopenadd = File::open(fileopen.pathfile).expect("file not present");
        let fileread = BufReader::new(fileopenadd);
        let mut stringsec: Vec<String> = Vec::new();
        for i in fileread.lines() {
            let line = i.expect("file not present");
            if !line.starts_with(">") {
                stringsec.push(line);
            }
        }

        let chopseq = chop(stringsec).unwrap();
        let vectensor = tensorchop(chopseq).unwrap();
        Ok(vectensor)
    }
}

pub fn chop(inputvec: Vec<String>) -> Result<Vec<String>, Box<dyn Error>> {
    let mut veclone: Vec<String> = Vec::new();
    let valuemax = inputvecclone
        .iter()
        .map(|x| x.len())
        .collect::<Vec<_>>()
        .iter()
        .max()
        .unwrap();
    let minvalue = inputvec
        .iter()
        .map(|x| x.len())
        .collect::<Vec<_>>()
        .iter()
        .min()
        .unwrap();
    let mutualvalue = valuemax + minvalue / 2usize;
    for i in inputvec.iter() {
        if i.len() < mutualvalue {
            let stringadd = mutualvalue - i.len();
            let mut valueadd: Vec<String> = Vec::new();
            for i in 0..stringadd {
                valueadd.push("A")
            }
            veclone.push(format!("{}{}", i, valueadd.concat().to_string()));
        } else if i.len() > mutualvalue {
            let valueadd = i[0..mutualvalue - i.len()].to_string();
            veclone.push(valueadd);
        }
    }
    Ok(veclone)
}

pub fn tensorchop(veclone: Vec<String>) -> Result<Tensor<f64>, Box<dyn Error>> {
    let mut vectensor: Vec<Tensor<f64>> = Vec::new();
    for i in veclone.iter() {
        let mut valuestring: Vec<f64> = Vec::new();
        let valuchars = i.chars.collect::<Vec<_>>();
        for i in valuchars {
            match i {
                'A' => valuchars.push(vec![1.0, 1.0, 0.0, 0.0]),
                'T' => valuchars.push(vec![0.0, 1.0, 0.0, 0.0]),
                'G' => valuchars.push(vec![0.0, 0.0, 1.0, 0.0]),
                'C' => valuchars.push(vec![0.0, 0.0, 0.0, 1.0]),
                _ => continue,
            }
        }
        let untangletensor = valuestring.iter().flatten().cloned().collect::<Vec<f64>>();
        let tensor = Tensor::from_vec(&untangletensor);
        vectensor.push(tensor);
    }
    let returntensor = Tensor::stack(&vectensor, 0);
}
