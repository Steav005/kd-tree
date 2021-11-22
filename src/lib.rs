struct Tree<T: Sized, V: Sized, const SIZE: usize, const DIM: usize> {
    nodes: [Node<T, V, DIM>; SIZE],
}

impl<T: Sized + PartialOrd + PartialEq, V: Sized, const SIZE: usize, const DIM: usize>
    Tree<T, V, SIZE, DIM>
{
    pub fn sort(values: [([T; DIM], V); SIZE]) -> [([T; DIM], V); SIZE] {
        //Add variable for final index
        let mut values: [_; SIZE] = values.map(|(a, b)| (a, b, 0usize));
        //Call Recursive Sort function
        Self::rec_sort(&mut values, 0, 0);

        //Apply final index by sorting by this index
        values.sort_by(|(_, _, a), (_, _, b)| {
            // Verify inequality of indices
            assert_ne!(a, b);
            a.cmp(b)
        });

        //Remove index and return
        values.map(|(a, b, _)| (a, b))
    }

    fn rec_sort(values: &mut [([T; DIM], V, usize)], dim: usize, index: usize) {
        //Check dimension
        let dim = dim % DIM;

        // Sort by current dimension
        values.sort_by(|(a, _, _), (b, _, _)| a[dim].partial_cmp(&b[dim]).unwrap());

        if values.len() == 1 {
            values[0].2 = index;
            return;
        } else if values.len() == 2 {
            values[1].2 = index;
            Self::rec_sort(&mut values[..1], dim + 1, Self::left(index));
            return;
        }
        let mid = (values.len()) / 2;

        let (left, rest) = values.split_at_mut(mid);
        let (mid, right) = rest.split_at_mut(1);

        mid.get_mut(0).unwrap().2 = index;
        Self::rec_sort(left, dim + 1, Self::left(index));
        Self::rec_sort(right, dim + 1, Self::right(index));
    }

    fn search<D>(&self, func: D) -> &Node<T, V, DIM>
    where
        D: Fn(&[T], &[T]) -> T,
    {
        let node = self.nodes.first().unwrap();
        let index = 0usize;
        loop {
        }

        todo!()
    }

    fn parent(cur_index: usize) -> usize {
        (cur_index + 1) / 2 - 1
    }

    fn left(cur_index: usize) -> usize {
        (cur_index + 1) * 2 - 1
    }

    fn right(cur_index: usize) -> usize {
        (cur_index + 1) * 2
    }
}

struct Node<T: Sized, V: Sized, const DIM: usize> {
    dim: usize,
    val: [T; DIM],
    v: V,
}

#[cfg(test)]
mod tests {
    use crate::Tree;
    use std::ops::{Add, Mul};

    fn euclid<T>(left: &[T], right: &[T]) -> T
    where
        T: Default + Add<Output = T> + Mul<Output = T> + Copy,
    {
        left.iter()
            .zip(right.iter())
            .fold(T::default(), |p, (a, b)| p + (*a + *b) * (*a + *b))
    }

    #[test]
    fn fixed_test() {
        let test = Tree::sort([
            ([5, 4], 0),
            ([2, 3], 1),
            ([8, 1], 2),
            ([9, 6], 3),
            ([7, 2], 4),
            ([4, 7], 5),
        ]);
        let correct = [
            ([7, 2], 4),
            ([5, 4], 0),
            ([9, 6], 3),
            ([2, 3], 1),
            ([4, 7], 5),
            ([8, 1], 2),
        ];

        assert_eq!(test, correct)
    }

    #[test]
    fn search() {
        todo!()
    }
}
