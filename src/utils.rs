use indexmap::IndexSet;
use std::collections::HashMap;
use std::fmt::Display;
use std::hash::Hash;

/// Error raised when no common supremum exists.
#[derive(Debug)]
pub struct SupremumError;

impl Display for SupremumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("supremum does not exist for the two chains")
    }
}

/// Compute the supremum of two chains `x` and `y`.
///
/// The type `T` must be:
/// * `Eq + Hash`  – for set / map membership
/// * `Clone`      – because we have to copy the element into the output set
///
/// If the supremum exists it is returned in the same stable order in which
/// elements are first picked from either input.
/// Otherwise a `SupremumError` is returned.
///
/// This is a direct transliteration of the original C++ logic, line-for-line.
pub fn supremum<T>(x: &[T], y: &[T]) -> Result<Vec<T>, SupremumError>
where
    T: Eq + Hash + Clone,
{
    // Remember where every element appears in each list.
    let mut pos_x = HashMap::<&T, usize>::with_capacity(x.len());
    let mut pos_y = HashMap::<&T, usize>::with_capacity(y.len());

    for (idx, elem) in x.iter().enumerate() {
        pos_x.insert(elem, idx);
    }
    for (idx, elem) in y.iter().enumerate() {
        pos_y.insert(elem, idx);
    }

    // Ordered-set that becomes our answer.
    let mut result: IndexSet<T> = IndexSet::with_capacity(x.len() + y.len());

    let (mut i, mut j) = (0usize, 0usize);
    const INF: usize = usize::MAX;

    while i < x.len() || j < y.len() {
        // Skip already-selected elements.
        while i < x.len() && result.contains(&x[i]) {
            i += 1;
        }
        while j < y.len() && result.contains(&y[j]) {
            j += 1;
        }
        if i >= x.len() && j >= y.len() {
            break;
        }

        // Case 1: both cursors point at the same element.
        if i < x.len() && j < y.len() && x[i] == y[j] {
            if pos_y[&x[i]] < j {
                return Err(SupremumError);
            }
            result.insert(x[i].clone());
            i += 1;
            j += 1;
            continue;
        }

        // Otherwise we may have to pick from one list.
        let mut cand_x = INF;
        let mut cand_y = INF;

        if i < x.len() {
            if let Some(&p) = pos_y.get(&x[i]) {
                if p >= j {
                    cand_x = p;
                }
            }
        }
        if j < y.len() {
            if let Some(&p) = pos_x.get(&y[j]) {
                if p >= i {
                    cand_y = p;
                }
            }
        }

        // Whichever candidate is “safe” to pick now wins.
        if i < x.len() && cand_x == INF {
            result.insert(x[i].clone());
            i += 1;
            continue;
        }
        if j < y.len() && cand_y == INF {
            result.insert(y[j].clone());
            j += 1;
            continue;
        }

        if cand_x <= cand_y {
            if pos_y[&x[i]] < j {
                return Err(SupremumError);
            }
            result.insert(x[i].clone());
            i += 1;
        } else {
            if pos_x[&y[j]] < i {
                return Err(SupremumError);
            }
            result.insert(y[j].clone());
            j += 1;
        }
    }

    Ok(result.into_iter().collect())
}
