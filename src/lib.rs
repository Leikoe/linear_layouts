use std::collections::HashMap;

use algebra_ast::{BinaryOp, Expr, Type};
use indexmap::IndexMap;

mod algebra_ast;

fn is_power_of_two(n: u32) -> bool {
    // Exclude non‐positive numbers, then use the bit‐trick on the positive range.
    (n & (n - 1)) == 0
}

type BasesT = IndexMap<String, Vec<Vec<u32>>>;

/// A linear layout mapping from hardware indices to logical tensor indices.
/// Each input dimension has a list of “basis vectors” (one per bit of the
/// dimension), each of which is a vector of integers giving the contribution
/// to each output dimension.  Under the hood, this is a linear map over GF(2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearLayout {
    /// bases[in_dim] = Vec of basis vectors; each basis is a Vec<i32> of length
    /// equal to the number of output dims.
    bases: BasesT,
    /// out_dims[out_dim] = size (must be a power of two for surjective layouts).
    out_dims: IndexMap<String, u32>,
    /// Cached surjective flag.
    surjective: bool,
}

impl LinearLayout {
    fn from_parts(bases: BasesT, out_dims: IndexMap<String, u32>, surjective: bool) -> Self {
        Self {
            bases,
            out_dims,
            surjective,
        }
    }

    // fn from_parts_infer_out_dim_sizes<N: Into<String>>(bases: BasesT, out_dim_names: &[N]) -> Self
    // where
    //     String: From<&N>,
    // {
    //     let out_dim_names: Vec<String> = out_dim_names.into_iter().map(|n| n.into()).collect();
    //     let mut out_dims = HashMap::from_iter(out_dim_names.iter().cloned().map(|n| (n, 1)));

    //     for (in_dim, in_dim_bases) in &bases {
    //         for basis in in_dim_bases {
    //             for i in 0..basis.len() {
    //                 let size: &mut u32 = out_dims.get_mut(&out_dim_names[i]).unwrap();
    //                 *size = (*size).max(basis[i].next_power_of_two())
    //             }
    //         }
    //     }

    //     let ll = LinearLayout::from_parts(bases, out_dims, true);
    //     let err = ll.check_invariants(true);
    //     err.expect("layout didn't respect invariants");
    //     ll
    // }

    pub fn empty() -> Self {
        Self::from_parts(IndexMap::new(), IndexMap::new(), true)
    }

    pub fn strided_1d(
        size: u32,
        stride: u32,
        in_dim: impl Into<String>,
        out_dim: impl Into<String>,
    ) -> Self {
        if size == 0 {
            return LinearLayout::empty();
        }

        assert!(is_power_of_two(size));
        let mut bases = Vec::new();
        let mut i = 1;
        while i < size {
            bases.push(vec![i * stride]);
            i *= 2;
        }
        let requires_surjective = stride == 1;

        LinearLayout::from_parts(
            IndexMap::from([(in_dim.into(), bases)]),
            IndexMap::from([(out_dim.into(), stride * size)]),
            requires_surjective,
        )
    }

    pub fn identity_1d(size: u32, in_dim: impl Into<String>, out_dim: impl Into<String>) -> Self {
        Self::strided_1d(size, 1, in_dim, out_dim)
    }

    pub fn get_in_dim_size_log2(&self, in_dim: impl Into<String>) -> u32 {
        self.bases
            .get(&in_dim.into())
            .expect("couldn't find out_dim in out_dims")
            .len() as u32
    }

    pub fn get_in_dim_size(&self, in_dim: impl Into<String>) -> u32 {
        1 << self.get_in_dim_size_log2(in_dim)
    }

    pub fn get_out_dim_size_log2(&self, out_dim: impl Into<String>) -> u32 {
        *(self
            .out_dims
            .get(&out_dim.into())
            .expect("couldn't find out_dim in out_dims"))
    }

    pub fn get_out_dim_size(&self, out_dim: impl Into<String>) -> u32 {
        1 << self.get_out_dim_size_log2(out_dim)
    }

    pub fn get_in_dim_names(&self) -> Vec<String> {
        self.bases.keys().cloned().collect()
    }

    pub fn get_out_dim_names(&self) -> Vec<String> {
        self.out_dims.keys().cloned().collect()
    }

    fn check_invariants(&self, require_surjective: bool) -> Option<String> {
        for (in_dim, in_dim_bases) in &self.bases {
            for basis in in_dim_bases {
                if basis.iter().any(|x| *x < 0) {
                    return Some(format!(
                        "Invalid bases passed to LinearLayout. Expected all basis values to be non-negative, but found a negative value for in dimension '{}'. Full list of bases: {:?}\n",
                        in_dim, self
                    ));
                }
            }
        }

        None
    }

    fn get_out_dim_index(&self, out_dim_name: &String) -> usize {
        self.out_dims.get_index_of(out_dim_name).unwrap()
    }

    fn get_basis(&self, in_dim_name: &String, pos: u32) -> &Vec<u32> {
        &self.bases.get(in_dim_name).unwrap()[pos as usize]
    }

    fn get_basis_value(&self, in_dim_name: &String, pos: u32, out_dim_name: &String) -> u32 {
        self.get_basis(in_dim_name, pos)[self.get_out_dim_index(out_dim_name)]
    }

    fn apply<'a>(&self, indices: &HashMap<String, Expr>) -> HashMap<String, Expr> {
        let mut out_indices = HashMap::new();
        for out_dim_name in self.out_dims.keys() {
            out_indices.insert(out_dim_name.clone(), Expr::u32_const(0));
        }

        for (in_dim_name, idx) in indices.iter() {
            let n_bits = self.get_in_dim_size_log2(in_dim_name);
            for i in 0..n_bits {
                let bit = Expr::binary(BinaryOp::And, idx.clone(), Expr::u32_const(1 << i));
                let bit_is_zero = Expr::binary(BinaryOp::Eq, bit, Expr::u32_const(0));
                for (out_dim_name, out_idx) in out_indices.iter_mut() {
                    let basis = self.get_basis_value(in_dim_name, i, out_dim_name);
                    if basis == 0 {
                        continue;
                    }
                    let basis_value = Expr::u32_const(basis);
                    *out_idx = Expr::binary(
                        BinaryOp::Xor,
                        out_idx.clone(),
                        Expr::select(bit_is_zero.clone(), Expr::u32_const(0), basis_value),
                    );
                }
            }
        }

        out_indices
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use algebra_ast::*;

    #[test]
    fn test_empty() {
        let layout = LinearLayout::empty();
        assert!(layout.bases.is_empty());
        assert!(layout.get_in_dim_names().is_empty());
        assert!(layout.get_out_dim_names().is_empty());
    }

    #[test]
    fn test_identity_1d() {
        let layout = LinearLayout::identity_1d(32, "ins", "outs");
        assert_eq!(
            layout,
            LinearLayout::from_parts(
                IndexMap::from([(
                    "ins".to_owned(),
                    vec![vec![1], vec![2], vec![4], vec![8], vec![16]]
                )]),
                IndexMap::new(),
                true
            )
        );
        assert_eq!(layout.bases.get("ins").unwrap().len(), 5);
        assert_eq!(layout.get_out_dim_size("outs"), 32);
    }

    #[test]
    fn test_identity_1d_size1() {
        let layout = LinearLayout::identity_1d(1, "ins", "outs");
        assert_eq!(layout.bases.get("ins").unwrap().len(), 0);
        assert_eq!(layout.get_out_dim_size_log2("outs"), 0);
    }

    #[test]
    fn test_apply() {
        let ll = LinearLayout::identity_1d(32, "ins", "outs");
        let indices = HashMap::from([("ins".to_owned(), Expr::var("i", Type::U32))]);
        let idx = ll.apply(&indices);

        let out_idx = idx["outs"].clone();
        let result = rewrite_fixpoint(out_idx, &RULES);
        println!("{}", result);
    }

    // #[test]
    // fn test_zeros1d() {
    //     let layout = LinearLayout::zeros1d(32, "ins", "outs", None);
    //     let vecs = layout.bases.get("ins").unwrap();
    //     assert_eq!(vecs.len(), 5);
    //     for v in vecs {
    //         assert_eq!(v[0], 0);
    //     }
    //     assert_eq!(layout.get_out_dim_size("outs"), 1);
    // }

    // #[test]
    // fn test_multiply_identity() {
    //     let l1 = LinearLayout::identity1d(16, "in", "out");
    //     let l2 = LinearLayout::identity1d(32, "in", "out");
    //     let prod = l1.compose(&l2);
    //     // Expect identity1d of size 32.
    //     let expected = LinearLayout::identity1d(32, "in", "out");
    //     assert_eq!(prod, expected);
    // }

    // #[test]
    // fn test_multiply_disjoint() {
    //     let l1 = LinearLayout::identity1d(32, "in1", "out1");
    //     let l2 = LinearLayout::identity1d(16, "in2", "out2");
    //     let prod = l1.compose(&l2);
    //     // bases:
    //     // in1: [{1,0}, {2,0}, {4,0}, {8,0}, {16,0}]
    //     // in2: [{0,1}, {0,2}, {0,4}, {0,8}]
    //     let mut expected_bases = BTreeMap::new();
    //     expected_bases.insert(
    //         "in1".to_string(),
    //         vec![vec![1, 0], vec![2, 0], vec![4, 0], vec![8, 0], vec![16, 0]],
    //     );
    //     expected_bases.insert(
    //         "in2".to_string(),
    //         vec![vec![0, 1], vec![0, 2], vec![0, 4], vec![0, 8]],
    //     );
    //     let mut expected_out_dims = BTreeMap::new();
    //     expected_out_dims.insert("out1".to_string(), 32);
    //     expected_out_dims.insert("out2".to_string(), 16);
    //     let expected = LinearLayout {
    //         bases: expected_bases,
    //         out_dims: expected_out_dims,
    //         surjective: true,
    //     };
    //     assert_eq!(prod, expected);
    // }

    // #[test]
    // fn test_multiply_by_empty() {
    //     let empty = LinearLayout::empty();
    //     let l = LinearLayout::identity1d(32, "in", "out");
    //     let prod = empty.compose(&l);
    //     assert_eq!(prod, l);
    // }

    // #[test]
    // fn test_multiply_by_zeros() {
    //     let l1 = LinearLayout::identity1d(8, "in", "out");
    //     let l2 = LinearLayout::zeros1d(16, "in", "out", None);
    //     let prod = l1.compose(&l2);
    //     // Expect bases: in: [{1},{2},{4},{0},{0},{0},{0}] since zeros override.
    //     let mut expected_bases = BTreeMap::new();
    //     expected_bases.insert(
    //         "in".to_string(),
    //         vec![
    //             vec![1],
    //             vec![2],
    //             vec![4],
    //             vec![0],
    //             vec![0],
    //             vec![0],
    //             vec![0],
    //         ],
    //     );
    //     let mut expected_out_dims = BTreeMap::new();
    //     expected_out_dims.insert("out".to_string(), 16);
    //     let expected = LinearLayout {
    //         bases: expected_bases,
    //         out_dims: expected_out_dims,
    //         surjective: false,
    //     };
    //     assert_eq!(prod, expected);
    // }

    // #[test]
    // fn test_apply() {
    //     let mut bases = BTreeMap::new();
    //     bases.insert("in1".to_string(), vec![vec![4, 2], vec![2, 1], vec![1, 0]]);
    //     bases.insert("in2".to_string(), vec![vec![1, 2], vec![2, 1]]);
    //     let out_dims = vec![("out1".to_string(), 8), ("out2".to_string(), 4)];
    //     let layout = LinearLayout::with_bases_and_out_dims(bases, out_dims, false).unwrap();
    //     // apply {in1=1, in2=0} → out: (4,2)
    //     let res = layout.apply(&[("in2".to_string(), 0), ("in1".to_string(), 1)]);
    //     let mut map = BTreeMap::new();
    //     for (d, v) in res {
    //         map.insert(d, v);
    //     }
    //     assert_eq!(map["out1"], 4);
    //     assert_eq!(map["out2"], 2);
    //     // apply {in2=1, in1=0} → (1,2)
    //     let res2 = layout.apply(&[("in2".to_string(), 1), ("in1".to_string(), 0)]);
    //     let mut map2 = BTreeMap::new();
    //     for (d, v) in res2 {
    //         map2.insert(d, v);
    //     }
    //     assert_eq!(map2["out1"], 1);
    //     assert_eq!(map2["out2"], 2);
    // }

    // #[test]
    // fn test_transpose_outs() {
    //     let l = LinearLayout::identity1d(32, "in1", "out1")
    //         .compose(&LinearLayout::identity1d(16, "in2", "out2"));
    //     let trans = l.transpose_outs(&["out2".to_string(), "out1".to_string()]);
    //     assert_eq!(
    //         trans.out_dim_names(),
    //         vec![&"out2".to_string(), &"out1".to_string()]
    //     );
    //     // Check bases permuted.
    //     let b_in1 = trans.bases.get("in1").unwrap();
    //     assert_eq!(b_in1[0], vec![0, 1]); // was [1,0]
    // }

    // #[test]
    // fn test_transpose_ins() {
    //     let l = LinearLayout::identity1d(32, "in1", "out1")
    //         .compose(&LinearLayout::identity1d(16, "in2", "out2"));
    //     let trans = l.transpose_ins(&["in2".to_string(), "in1".to_string()]);
    //     assert_eq!(
    //         trans.in_dim_names(),
    //         vec![&"in2".to_string(), &"in1".to_string()]
    //     );
    //     let b_in2 = trans.bases.get("in2").unwrap();
    //     assert_eq!(b_in2[0], vec![0, 1]); // was [1,0]
    // }

    // #[test]
    // fn test_get_num_consecutive_in_out() {
    //     let id4 = LinearLayout::identity1d(4, "in", "out");
    //     assert_eq!(id4.get_num_consecutive_in_out(), 4);
    //     let prod = LinearLayout::identity1d(4, "in1", "out")
    //         .compose(&LinearLayout::identity1d(8, "in2", "out"));
    //     assert_eq!(prod.get_num_consecutive_in_out(), 4);
    //     let zeros = LinearLayout::zeros1d(4, "in", "out1", None)
    //         .compose(&LinearLayout::identity1d(4, "in", "out2"));
    //     assert_eq!(zeros.get_num_consecutive_in_out(), 1);
    // }

    // #[test]
    // fn test_sublayout_and_zero() {
    //     let mut bases = BTreeMap::new();
    //     bases.insert("in1".to_string(), vec![vec![1, 0], vec![0, 1]]);
    //     bases.insert("in2".to_string(), vec![vec![0, 1]]);
    //     let layout = LinearLayout::with_bases_and_out_dims(
    //         bases,
    //         vec![("out1".to_string(), 2), ("out2".to_string(), 2)],
    //         true,
    //     )
    //     .unwrap();
    //     let sub = layout.sublayout(&vec!["in1".to_string()], &vec!["out1".to_string()]);
    //     // Expect a single-in→single-out basis [1],[0] etc.
    //     let b = sub.bases.get("in1").unwrap();
    //     assert_eq!(b.len(), 2);
    //     assert_eq!(b[0], vec![1]);
    //     assert_eq!(b[1], vec![0]);
    //     assert!(layout.sublayout_is_zero(&vec!["in1".to_string()], &vec!["out2".to_string()]));
    // }

    // #[test]
    // fn test_free_variable_masks() {
    //     let id4 = LinearLayout::identity1d(4, "in", "out");
    //     let masks = id4.get_free_variable_masks();
    //     assert_eq!(masks[&"in".to_string()], 0);
    //     let zeros16 = LinearLayout::zeros1d(16, "in", "out", None);
    //     let masks2 = zeros16.get_free_variable_masks();
    //     assert_eq!(masks2[&"in".to_string()], 0b1111);
    //     let mix = id4.compose(&zeros16).compose(&id4).compose(&zeros16);
    //     let masks3 = mix.get_free_variable_masks();
    //     assert_eq!(masks3[&"in".to_string()], 0b100110);
    // }

    // #[test]
    // fn test_divide_left_basic() {
    //     let b = LinearLayout::identity1d(8, "in", "out");
    //     let c = LinearLayout::zeros1d(16, "in", "out", None);
    //     let a = b.compose(&c);
    //     let got_c = LinearLayout::divide_left(&a, &b).unwrap();
    //     assert_eq!(got_c, c);
    //     let got_b = LinearLayout::divide_left(&c.compose(&b), &c).unwrap();
    //     assert_eq!(got_b, b);
    // }

    // #[test]
    // fn test_column_action_apply_layout() {
    //     let mut bases = BTreeMap::new();
    //     bases.insert("in".to_string(), vec![vec![1], vec![2], vec![4]]);
    //     let layout =
    //         LinearLayout::with_bases_and_out_dims(bases, vec![("out".to_string(), 8)], true)
    //             .unwrap();
    //     let ca = ColumnAction::new(vec![2, 0, 1], "in", 3);
    //     let trans = ca.apply_layout(&layout);
    //     let vecs = trans.bases.get("in").unwrap();
    //     assert_eq!(vecs, &vec![vec![4], vec![1], vec![2]]);
    //     let ca2 = ColumnAction::new(vec![1, 0], "in", 3);
    //     let trans2 = ca2.apply_layout(&layout);
    //     let vecs2 = trans2.bases.get("in").unwrap();
    //     assert_eq!(vecs2, &vec![vec![2], vec![1]]);
    // }

    // #[test]
    // fn test_column_action_apply_values() {
    //     let values: Vec<usize> = (1..=8).collect();
    //     let ca = ColumnAction::new(vec![2, 0, 1], "register", 3);
    //     let permuted = ca.apply_values(&values);
    //     assert_eq!(permuted, vec![1, 5, 2, 6, 3, 7, 4, 8]);
    //     let ca2 = ColumnAction::new(vec![2, 1], "register", 3);
    //     let permuted2 = ca2.apply_values(&values);
    //     assert_eq!(permuted2, vec![1, 5, 3, 7]);
    // }
}
