use std::{slice, sync::Arc, vec};
pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Copy + Clone + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length: length,
        }
    }

    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }

    pub fn better_slice(&self, slices: &[(usize, usize)]) -> Self {
        assert_eq!(slices.len(), self.shape.len(), "Number of slices must match the number of dimensions.");

        let mut new_shape = Vec::new();
        let mut new_length = 1;

        for (i, &(start, end)) in slices.iter().enumerate() {
            assert!(start <= end && end <= self.shape[i], "Slice indices are out of bounds.");

            new_shape.push(end - start);
            new_length *= end - start;

            // 计算新偏移
            //new_offset += start * self.calculate_stride(i);
        }
        
        let new_data = self.extract_data(&slices);
        let new_data_arc = Arc::new(new_data.into_boxed_slice());

        Tensor {
            data: new_data_arc,
            shape: new_shape,
            offset: self.offset,
            length: new_length,
        }
    }

    fn calculate_stride(&self, dim: usize) -> usize {
        let stride: usize = self.shape.iter().skip(dim + 1).product();
        stride
    }

    fn extract_data(&self, slices: &[(usize, usize)]) -> Vec<T> {
        let mut result = Vec::new();
        let mut indices = vec![0; slices.len()];

        let total_elements = slices.iter().map(|&(start, end)| end - start).product::<usize>();

        for _ in 0..total_elements {
            let mut index = self.offset;
            for (dim, &(start, _)) in slices.iter().enumerate() {
                index += (indices[dim] + start) * self.calculate_stride(dim);
            }

            result.push(self.data[index]);

            for dim in (0..slices.len()).rev() {
                indices[dim] += 1;
                if indices[dim] < slices[dim].1 - slices[dim].0 {
                    break;
                }
                indices[dim] = 0; 
            }
        }

        result
    }


}

// Some helper functions for testing and debugging
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();
        
        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self){
        println!("shape: {:?}, offset: {}, length: {}", self.shape, self.offset, self.length);
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}


#[test]
fn test_better_slice(){
    let data = vec![1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0];
    let tensor = Tensor::<f32>::new(data, &vec![2,3,2]);
    tensor.print();
    /*
    shape: [2, 3, 2], offset: 0, length: 12
    [[1.0, 2.0]
    [3.0, 4.0]
    [5.0, 6.0]],
    [[7.0, 8.0]
    [9.0, 10.0]
    [11.0, 12.0]]
     */

    //倘若要对最后面的维度slice，直接调用老slice方法并不能，它只是按照start和offset计算data，然后把新的tensor重塑成shape的形状
    let old_slice = tensor.slice(0,&vec![2,3,1] );
    old_slice.print();
    /*
    shape: [1, 3, 2], offset: 0, length: 6
    [1.0, 2.0]
    [3.0, 4.0]
    [5.0, 6.0]
     */
    //一个新的实现，可以让我们更便捷切片...
    let tuple_vec: Vec<(usize, usize)> = vec![(0, 2), (0, 3), (0, 1)];
    let new_slice = tensor.better_slice(&tuple_vec);
    new_slice.print();
    /*
    shape: [2, 3, 1], offset: 0, length: 6
    [1.0]
    [3.0]
    [5.0]
    [7.0]
    [9.0]
    [11.0]
     */

    assert!(new_slice.close_to(&Tensor::<f32>::new(vec![1.0,3.0,5.0,7.0,9.0,11.0],&vec![2, 3, 1]), 1e-3));

}