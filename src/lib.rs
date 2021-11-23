//#![no_std]
use core::cmp::Ordering;

struct Tree<T: Sized, V: Sized, const SIZE: usize, const DIM: usize, const MAX_LEVEL: usize> {
    nodes: [Node<T, V, DIM>; SIZE],
}

impl<T: Sized, V: Sized, const SIZE: usize, const DIM: usize, const MAX_LEVEL: usize>
    Tree<T, V, SIZE, DIM, MAX_LEVEL>
{
    pub const fn new(nodes: [Node<T, V, DIM>; SIZE]) -> Tree<T, V, SIZE, DIM, MAX_LEVEL> {
        Tree { nodes }
    }
}

impl<
        T: Sized + PartialOrd + PartialEq + core::ops::Sub<Output = T> + Copy,
        V: Sized,
        const SIZE: usize,
        const DIM: usize,
        const MAX_LEVEL: usize,
    > Tree<T, V, SIZE, DIM, MAX_LEVEL>
{
    pub fn search<D>(&self, point: &[T; DIM], distance: D) -> &Node<T, V, DIM>
    where
        D: Fn(&[T; DIM], &[T; DIM]) -> T,
    {
        let mut node: &Node<T, V, DIM> = &self
            .nodes
            .first()
            .expect("This can not fail, expect if the Tree got no nodes: SIZE == 0");
        let mut index = 0usize;
        let mut level = 0usize;
        // Store which child nodes where visited for the Node on the index/level
        let mut visited: [Visited; MAX_LEVEL] = [Visited::None; MAX_LEVEL];
        //Initialise best node/distance with root node
        let mut best_distance: T = distance(&node.val, &point);
        let mut best_node: &Node<T, V, DIM> = node;

        loop {
            // Get to leaf based on comparison of only a single dimension per level
            self.search_down(point, &mut node, &mut index, &mut level, &mut visited);

            // Go up until we either reach the top or we go down by one
            // Should we go down by one, we will need to go to the best fit leaf of the current subtree
            self.search_up(
                point,
                &mut node,
                &mut index,
                &mut level,
                &mut visited,
                &mut best_distance,
                &mut best_node,
                &distance,
            );

            // Should we have reached level 0, we are finished
            // as search_up should go down by one should we still need to search a subtree
            if level == 0 {
                break;
            }
        }

        best_node
    }

    fn search_down<'a>(
        &'a self,
        point: &[T; DIM],
        node: &mut &'a Node<T, V, DIM>,
        index: &mut usize,
        level: &mut usize,
        visited: &mut [Visited; MAX_LEVEL],
    ) {
        //Get to leaf node
        loop {
            let dim = *level % DIM;

            //If the left node is not reachable we are at a leaf node
            if left(index) >= SIZE {
                //Set visited for this level to ALL because there are no more child nodes
                visited[*level] = Visited::All;
                return;
            }
            //Decide where to go
            *index = match point[dim].partial_cmp(&node.val[dim]) {
                Some(Ordering::Equal) | Some(Ordering::Less) => left(index),
                Some(Ordering::Greater) => right(index),
                _ => panic!(),
            };

            // If the index is to big we choose the right node
            // Make sure it exists, if not go to the left node instead
            if *index >= SIZE {
                *index -= 1;
            }

            // Set next node
            *node = &self.nodes.get(*index).unwrap();

            // Increase level for next node
            *level += 1;
        }
    }

    fn search_up<'a, D>(
        &'a self,
        point: &[T; DIM],
        node: &mut &'a Node<T, V, DIM>,
        index: &mut usize,
        level: &mut usize,
        visited: &mut [Visited; MAX_LEVEL],
        best_distance: &mut T,
        best_node: &mut &'a Node<T, V, DIM>,
        distance: D,
    ) where
        D: Fn(&[T; DIM], &[T; DIM]) -> T,
    {
        loop {
            let dim = *level % DIM;

            // Check if current node is a closer node
            let candidate: T = distance(&node.val, &point);
            if candidate < *best_distance {
                *best_distance = candidate;
                *best_node = node;
            }

            // Determine where to go? Up, BottomLeft or BottomRight
            let dir: Direction = match visited[*level] {
                Visited::All => Direction::Up,
                Visited::Left => {
                    // Check if we even can go right
                    // and if its even possible for the right side to be nearer
                    if right(index) < SIZE && node.val[dim].sub(point[dim]) < *best_distance {
                        Direction::Right
                    } else {
                        Direction::Up
                    }
                }
                Visited::Right => {
                    // Check if we even can go left
                    // and if its even possible for the left side to be nearer
                    if left(index) < SIZE && point[dim].sub(node.val[dim]) < *best_distance {
                        Direction::Left
                    } else {
                        Direction::Up
                    }
                }
                _ => panic!("Unexpected state"),
            };

            match dir {
                Direction::Up => {
                    //Stop if we are at level 0
                    if *level == 0 {
                        return;
                    }

                    // Move up
                    let parent_index = parent(index);
                    *level -= 1;
                    // But update the visited status of the parent first
                    // This way we know which childs were already visited
                    match visited[*level] {
                        Visited::Left | Visited::Right => visited[*level] = Visited::All,
                        _ => {
                            if left(&parent_index) == *index {
                                visited[*level] = Visited::Left;
                            } else {
                                visited[*level] = Visited::Right;
                            }
                        }
                    }
                    *index = parent_index;
                    *node = self.nodes.get(*index).unwrap();
                }
                Direction::Left => {
                    *level += 1;
                    visited[*level] = Visited::None;
                    *index = left(index);
                    *node = self.nodes.get(*index).unwrap();
                    return;
                }
                Direction::Right => {
                    *level += 1;
                    visited[*level] = Visited::None;
                    *index = right(index);
                    *node = self.nodes.get(*index).unwrap();
                    return;
                }
            }
        }
    }
}

#[derive(Copy, Clone)]
enum Visited {
    None,
    Left,
    Right,
    All,
}

enum Direction {
    Up,
    Left,
    Right,
}

pub fn sort<
    T: Sized + PartialOrd + PartialEq + core::fmt::Debug,
    V: Sized,
    const SIZE: usize,
    const DIM: usize,
>(
    values: [([T; DIM], V); SIZE],
) -> [([T; DIM], V); SIZE] {
    //Add variable for final index
    let mut values: [_; SIZE] = values.map(|(a, b)| (a, b, 0usize));
    //Call Recursive Sort function
    rec_sort(&mut values, 0, 0);

    //Apply final index by sorting by this index
    values.sort_by(|(_, _, a), (_, _, b)| {
        // Verify inequality of indices
        assert_ne!(a, b);
        a.cmp(b)
    });

    //Remove index and return
    values.map(|(a, b, _)| (a, b))
}

fn rec_sort<T: Sized + PartialOrd + PartialEq, V: Sized, const DIM: usize>(
    values: &mut [([T; DIM], V, usize)],
    dim: usize,
    index: usize,
) {
    //Check dimension
    let dim = dim % DIM;

    // Sort by current dimension
    values.sort_by(|(a, _, _), (b, _, _)| a[dim].partial_cmp(&b[dim]).unwrap());

    if values.len() == 1 {
        values[0].2 = index;
        return;
    } else if values.len() == 2 {
        values[1].2 = index;
        rec_sort(&mut values[..1], dim + 1, left(&index));
        return;
    }
    let mid = (values.len()) / 2;

    let (left_slice, rest) = values.split_at_mut(mid);
    let (mid, right_slice) = rest.split_at_mut(1);

    mid.get_mut(0).unwrap().2 = index;
    rec_sort(left_slice, dim + 1, left(&index));
    rec_sort(right_slice, dim + 1, right(&index));
}

fn parent(cur_index: &usize) -> usize {
    (cur_index + 1) / 2 - 1
}

fn left(cur_index: &usize) -> usize {
    (cur_index + 1) * 2 - 1
}

fn right(cur_index: &usize) -> usize {
    (cur_index + 1) * 2
}

#[derive(Debug)]
struct Node<T: Sized, V: Sized, const DIM: usize> {
    val: [T; DIM],
    v: V,
}

#[cfg(test)]
mod tests {
    use crate::{sort, Node, Tree};
    use rand::Rng;
    use std::ops::{Add, Mul, Sub};

    fn euclid<T, const DIM: usize>(left: &[T; DIM], right: &[T; DIM]) -> T
    where
        T: Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy + core::fmt::Debug,
    {
        let res = left
            .iter()
            .zip(right.iter())
            .fold(T::default(), |p, (a, b)| p + (*a - *b) * (*a - *b));
        res
    }

    #[test]
    fn fixed_test() {
        let test = sort([
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
        let sorted = sort([
            ([5, 4], 0),
            ([2, 3], 1),
            ([8, 1], 2),
            ([9, 6], 3),
            ([7, 2], 4),
            ([4, 7], 5),
        ]);

        let nodes = sorted.map(|(p, v)| Node { val: p, v });
        //T, V, SIZE, DIM, MAX_LEVEL
        let tree = Tree::<i32, i32, 6, 2, 3>::new(nodes);

        assert_eq!(&tree.search(&[5, 3], &euclid).val, &[5, 4]);
        assert_eq!(&tree.search(&[9, 7], &euclid).val, &[9, 6]);
        assert_eq!(&tree.search(&[3, 9], &euclid).val, &[4, 7]);
        assert_eq!(&tree.search(&[3, 0], &euclid).val, &[2, 3]);
    }

    #[test]
    fn random_search() {
        let mut rng = rand::thread_rng();
        //const ITERATIONS: usize = 10;
        //const NODES_SIZE: usize = 150;
        //const MAX_LEVEL: usize = 8;
        //const SEARCHES: usize = 1000;
        const ITERATIONS: usize = 10;
        const NODES_SIZE: usize = 6;
        const MAX_LEVEL: usize = 3;
        const SEARCHES: usize = 1000;

        for i in 0..ITERATIONS {
            let values: [([f64; 3], i32); NODES_SIZE] =
                [0; NODES_SIZE].map(|_| ([rng.gen(), rng.gen(), rng.gen()], 0));
            let sorted = sort(values.clone());
            let nodes = sorted.map(|(p, v)| Node { val: p, v });
            let tree = Tree::<f64, i32, NODES_SIZE, 3, MAX_LEVEL>::new(nodes);

            let search_points: [[f64; 3]; SEARCHES] = [0; SEARCHES].map(|_| [rng.gen(), rng.gen(), rng.gen()]);
            let mut k = 0;
            for point in search_points{
                let closest: f64 = values.iter().map(|a| (a.clone(), euclid(&point, &a.0)))
                    .reduce(|(c_a, c_d), (n_a, n_d)| match c_d.partial_cmp(&n_d) {
                        Some(core::cmp::Ordering::Greater) => (n_a, n_d),
                        _ => (c_a, c_d),
                    })
                    .unwrap()
                    .1;
                let closest_node = tree.search(&point, euclid);
                println!("{}", k);
                k += 1;
                assert_eq!(closest, euclid(&closest_node.val, &point), "Tree: {:?}\nPoint: {:?}", sorted, point);
            }
        }
    }
}
