use std::fmt;
use std::rc::Rc;

/// =============================================
/// 1) AST Definitions (with helpers for all literal types)
/// =============================================

#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    F32(f32),
    F64(f64),
    Bool(bool),
}

impl Literal {
    pub fn ty(&self) -> Type {
        match self {
            Literal::I8(_) => Type::I8,
            Literal::U8(_) => Type::U8,
            Literal::I16(_) => Type::I16,
            Literal::U16(_) => Type::U16,
            Literal::I32(_) => Type::I32,
            Literal::U32(_) => Type::U32,
            Literal::F32(_) => Type::F32,
            Literal::F64(_) => Type::F64,
            Literal::Bool(_) => Type::Bool,
        }
    }

    fn value(&self) -> String {
        match self {
            Literal::I8(i) => i.to_string(),
            Literal::U8(u) => u.to_string(),
            Literal::I16(i) => i.to_string(),
            Literal::U16(u) => u.to_string(),
            Literal::I32(i) => i.to_string(),
            Literal::U32(u) => u.to_string(),
            Literal::F32(x) => x.to_string(),
            Literal::F64(x) => x.to_string(),
            Literal::Bool(b) => b.to_string(),
        }
    }

    pub fn cast(self, to_ty: Type) -> Literal {
        match (self, to_ty) {
            (Literal::I32(x), Type::F32) => Literal::F32(x as f32),
            (Literal::I32(x), Type::F64) => Literal::F64(x as f64),
            (Literal::F32(x), Type::I32) => Literal::I32(x as i32),
            (Literal::F64(x), Type::I32) => Literal::I32(x as i32),
            (lit, _) => lit,
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Bool(b) => write!(f, "{}", if *b { "true" } else { "false" }),
            lit => write!(f, "{}{}", lit.value(), lit.ty()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Const(pub Literal);

impl fmt::Display for Const {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
    Bool,
}

impl Type {
    pub fn suffix(&self) -> &'static str {
        match self {
            Type::I8 => "i8",
            Type::U8 => "u8",
            Type::I16 => "i16",
            Type::U16 => "u16",
            Type::I32 => "i32",
            Type::U32 => "u32",
            Type::F32 => "f32",
            Type::F64 => "f64",
            Type::Bool => "bool",
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.suffix())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Var {
    pub name: String,
    pub ty: Type,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    And,
    Xor,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl BinaryOp {
    pub fn associative(&self) -> bool {
        matches!(
            self,
            BinaryOp::Add | BinaryOp::Mul | BinaryOp::And | BinaryOp::Xor
        )
    }
    pub fn identity(&self) -> Option<Literal> {
        match self {
            BinaryOp::Add => Some(Literal::I32(0)),
            BinaryOp::Mul => Some(Literal::I32(1)),
            BinaryOp::And => Some(Literal::I32(!0i32)),
            BinaryOp::Xor => Some(Literal::I32(0)),
            _ => None,
        }
    }
    pub fn absorbing(&self) -> Option<Literal> {
        match self {
            BinaryOp::Mul => Some(Literal::I32(0)),
            BinaryOp::And => Some(Literal::I32(0)),
            _ => None,
        }
    }
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
        )
    }
    pub fn apply(&self, a: &Literal, b: &Literal) -> Option<Literal> {
        match (self, a, b) {
            (BinaryOp::Add, Literal::I32(x), Literal::I32(y)) => {
                Some(Literal::I32(x.wrapping_add(*y)))
            }
            (BinaryOp::Sub, Literal::I32(x), Literal::I32(y)) => {
                Some(Literal::I32(x.wrapping_sub(*y)))
            }
            (BinaryOp::Mul, Literal::I32(x), Literal::I32(y)) => {
                Some(Literal::I32(x.wrapping_mul(*y)))
            }
            (BinaryOp::And, Literal::I32(x), Literal::I32(y)) => Some(Literal::I32(x & y)),
            (BinaryOp::Xor, Literal::I32(x), Literal::I32(y)) => Some(Literal::I32(x ^ y)),
            (BinaryOp::Eq, Literal::I32(x), Literal::I32(y)) => Some(Literal::Bool(x == y)),
            (BinaryOp::Ne, Literal::I32(x), Literal::I32(y)) => Some(Literal::Bool(x != y)),
            (BinaryOp::Lt, Literal::I32(x), Literal::I32(y)) => Some(Literal::Bool(x < y)),
            (BinaryOp::Le, Literal::I32(x), Literal::I32(y)) => Some(Literal::Bool(x <= y)),
            (BinaryOp::Gt, Literal::I32(x), Literal::I32(y)) => Some(Literal::Bool(x > y)),
            (BinaryOp::Ge, Literal::I32(x), Literal::I32(y)) => Some(Literal::Bool(x >= y)),
            _ => None,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::And => "&",
            BinaryOp::Xor => "^",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Const(Const),
    Var(Var),

    Unary {
        op: UnaryOp,
        expr: Expr,
        ty: Type,
    },

    Binary {
        op: BinaryOp,
        left: Expr,
        right: Expr,
    },

    Cast {
        expr: Expr,
        to_ty: Type,
    },

    Select {
        cond: Expr,
        true_val: Expr,
        false_val: Expr,
    },

    Let {
        name: String,
        ty: Type,
        value: Expr,
        body: Expr,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr(Rc<ExprKind>);

impl Expr {
    /// Helpers for constructing literals of all supported types:
    pub fn i8_const(x: i8) -> Expr {
        Expr::const_from_lit(Literal::I8(x))
    }
    pub fn u8_const(x: u8) -> Expr {
        Expr::const_from_lit(Literal::U8(x))
    }
    pub fn i16_const(x: i16) -> Expr {
        Expr::const_from_lit(Literal::I16(x))
    }
    pub fn u16_const(x: u16) -> Expr {
        Expr::const_from_lit(Literal::U16(x))
    }
    pub fn i32_const(x: i32) -> Expr {
        Expr::const_from_lit(Literal::I32(x))
    }
    pub fn u32_const(x: u32) -> Expr {
        Expr::const_from_lit(Literal::U32(x))
    }
    pub fn f32_const(x: f32) -> Expr {
        Expr::const_from_lit(Literal::F32(x))
    }
    pub fn f64_const(x: f64) -> Expr {
        Expr::const_from_lit(Literal::F64(x))
    }
    pub fn bool_const(b: bool) -> Expr {
        Expr::const_from_lit(Literal::Bool(b))
    }

    pub fn const_from_lit(lit: Literal) -> Expr {
        Expr(Rc::new(ExprKind::Const(Const(lit))))
    }
    pub fn var(name: impl Into<String>, ty: Type) -> Expr {
        Expr(Rc::new(ExprKind::Var(Var {
            name: name.into(),
            ty,
        })))
    }
    pub fn unary(op: UnaryOp, expr: Expr) -> Expr {
        let ty = expr.ty();
        Expr(Rc::new(ExprKind::Unary { op, expr, ty }))
    }
    pub fn binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
        Expr(Rc::new(ExprKind::Binary { op, left, right }))
    }
    pub fn cast(expr: Expr, to_ty: Type) -> Expr {
        Expr(Rc::new(ExprKind::Cast { expr, to_ty }))
    }
    pub fn select(cond: Expr, tv: Expr, fv: Expr) -> Expr {
        assert_eq!(cond.ty(), Type::Bool, "select‐cond must be Bool");
        assert_eq!(tv.ty(), fv.ty(), "select branches must have same type");
        Expr(Rc::new(ExprKind::Select {
            cond,
            true_val: tv,
            false_val: fv,
        }))
    }
    pub fn let_in(name: impl Into<String>, ty: Type, value: Expr, body: Expr) -> Expr {
        Expr(Rc::new(ExprKind::Let {
            name: name.into(),
            ty,
            value,
            body,
        }))
    }

    pub fn ty(&self) -> Type {
        match &*self.0 {
            ExprKind::Const(Const(lit)) => lit.ty(),
            ExprKind::Var(Var { ty, .. }) => ty.clone(),
            ExprKind::Unary { ty, .. } => ty.clone(),
            ExprKind::Binary { op, left, .. } => {
                if op.is_comparison() {
                    Type::Bool
                } else {
                    left.ty()
                }
            }
            ExprKind::Cast { to_ty, .. } => to_ty.clone(),
            ExprKind::Select { true_val, .. } => true_val.ty(),
            ExprKind::Let { ty, .. } => ty.clone(),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &*self.0 {
            ExprKind::Const(Const(lit)) => write!(f, "{}", lit),

            ExprKind::Var(Var { name, .. }) => write!(f, "{}", name),

            ExprKind::Unary { op, expr, .. } => {
                let op_str = match op {
                    UnaryOp::Neg => "-",
                    UnaryOp::Not => "!",
                };
                write!(f, "({}{})", op_str, expr)
            }

            ExprKind::Binary { op, left, right } => {
                write!(f, "({} {} {})", left, op.name(), right)
            }

            ExprKind::Cast { expr, to_ty } => write!(f, "({} as {})", expr, to_ty),

            ExprKind::Select {
                cond,
                true_val,
                false_val,
            } => write!(f, "select({}, {}, {})", cond, true_val, false_val),

            ExprKind::Let {
                name,
                ty,
                value,
                body,
            } => write!(f, "(let {}: {} = {} in {})", name, ty, value, body),
        }
    }
}

/// =============================================
/// 2) Rewrite Framework
/// =============================================

pub type RewriteRule = fn(&Expr) -> Option<Expr>;

pub fn rewrite_once(expr: &Expr, rules: &[RewriteRule]) -> Expr {
    // Bottom‐up rewrite of children
    let rebuilt: Expr = match &*expr.0 {
        ExprKind::Const(_) | ExprKind::Var(_) => expr.clone(),

        ExprKind::Unary {
            op, expr: inner, ..
        } => {
            let new_inner = rewrite_once(inner, rules);
            Expr::unary(op.clone(), new_inner)
        }

        ExprKind::Binary { op, left, right } => {
            let new_left = rewrite_once(left, rules);
            let new_right = rewrite_once(right, rules);
            Expr::binary(op.clone(), new_left, new_right)
        }

        ExprKind::Cast { expr: inner, to_ty } => {
            let new_inner = rewrite_once(inner, rules);
            Expr::cast(new_inner, to_ty.clone())
        }

        ExprKind::Select {
            cond,
            true_val,
            false_val,
        } => {
            let new_cond = rewrite_once(cond, rules);
            let new_true = rewrite_once(true_val, rules);
            let new_false = rewrite_once(false_val, rules);
            Expr::select(new_cond, new_true, new_false)
        }

        ExprKind::Let {
            name,
            ty,
            value,
            body,
        } => {
            let new_value = rewrite_once(value, rules);
            let new_body = rewrite_once(body, rules);
            Expr::let_in(name.clone(), ty.clone(), new_value, new_body)
        }
    };

    for rule in rules {
        if let Some(new_node) = rule(&rebuilt) {
            return new_node;
        }
    }

    rebuilt
}

pub fn rewrite_fixpoint(expr: Expr, rules: &[RewriteRule]) -> Expr {
    let mut current = expr;
    loop {
        let next = rewrite_once(&current, rules);
        if format!("{}", next) == format!("{}", current) {
            return current;
        }
        current = next;
    }
}

/// =============================================
/// 3) Rewrite Rules
/// =============================================

fn fold_bin_literals(expr: &Expr) -> Option<Expr> {
    if let ExprKind::Binary { op, left, right } = &*expr.0 {
        if let (ExprKind::Const(Const(b_lit)), ExprKind::Const(Const(c_lit))) =
            (&*left.0, &*right.0)
        {
            if let Some(new_lit) = op.apply(b_lit, c_lit) {
                return Some(Expr::const_from_lit(new_lit));
            }
        }
    }
    None
}

fn fold_add_zero_left(expr: &Expr) -> Option<Expr> {
    if let ExprKind::Binary {
        op: BinaryOp::Add,
        left,
        right,
    } = &*expr.0
    {
        if let ExprKind::Const(Const(Literal::I32(0))) = &*left.0 {
            if right.ty() == Type::I32 {
                return Some(right.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::I16(0))) = &*left.0 {
            if right.ty() == Type::I16 {
                return Some(right.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::I8(0))) = &*left.0 {
            if right.ty() == Type::I8 {
                return Some(right.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::U32(0))) = &*left.0 {
            if right.ty() == Type::U32 {
                return Some(right.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::U16(0))) = &*left.0 {
            if right.ty() == Type::U16 {
                return Some(right.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::U8(0))) = &*left.0 {
            if right.ty() == Type::U8 {
                return Some(right.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::F32(x))) = &*left.0 {
            if *x == 0.0 && right.ty() == Type::F32 {
                return Some(right.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::F64(x))) = &*left.0 {
            if *x == 0.0 && right.ty() == Type::F64 {
                return Some(right.clone());
            }
        }
    }
    None
}

fn fold_add_zero_right(expr: &Expr) -> Option<Expr> {
    if let ExprKind::Binary {
        op: BinaryOp::Add,
        left,
        right,
    } = &*expr.0
    {
        if let ExprKind::Const(Const(Literal::I32(0))) = &*right.0 {
            if left.ty() == Type::I32 {
                return Some(left.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::I16(0))) = &*right.0 {
            if left.ty() == Type::I16 {
                return Some(left.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::I8(0))) = &*right.0 {
            if left.ty() == Type::I8 {
                return Some(left.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::U32(0))) = &*right.0 {
            if left.ty() == Type::U32 {
                return Some(left.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::U16(0))) = &*right.0 {
            if left.ty() == Type::U16 {
                return Some(left.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::U8(0))) = &*right.0 {
            if left.ty() == Type::U8 {
                return Some(left.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::F32(x))) = &*right.0 {
            if *x == 0.0 && left.ty() == Type::F32 {
                return Some(left.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::F64(x))) = &*right.0 {
            if *x == 0.0 && left.ty() == Type::F64 {
                return Some(left.clone());
            }
        }
    }
    None
}

fn fold_xor_zero_left(expr: &Expr) -> Option<Expr> {
    // (0 ^ x) => x for any integer type
    if let ExprKind::Binary {
        op: BinaryOp::Xor,
        left,
        right,
    } = &*expr.0
    {
        if let ExprKind::Const(Const(Literal::U32(0))) = &*left.0 {
            if right.ty() == Type::U32 {
                return Some(right.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::U16(0))) = &*left.0 {
            if right.ty() == Type::U16 {
                return Some(right.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::U8(0))) = &*left.0 {
            if right.ty() == Type::U8 {
                return Some(right.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::I32(0))) = &*left.0 {
            if right.ty() == Type::I32 {
                return Some(right.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::I16(0))) = &*left.0 {
            if right.ty() == Type::I16 {
                return Some(right.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::I8(0))) = &*left.0 {
            if right.ty() == Type::I8 {
                return Some(right.clone());
            }
        }
    }
    None
}

fn fold_xor_zero_right(expr: &Expr) -> Option<Expr> {
    // (x ^ 0) => x for any integer type
    if let ExprKind::Binary {
        op: BinaryOp::Xor,
        left,
        right,
    } = &*expr.0
    {
        if let ExprKind::Const(Const(Literal::U32(0))) = &*right.0 {
            if left.ty() == Type::U32 {
                return Some(left.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::U16(0))) = &*right.0 {
            if left.ty() == Type::U16 {
                return Some(left.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::U8(0))) = &*right.0 {
            if left.ty() == Type::U8 {
                return Some(left.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::I32(0))) = &*right.0 {
            if left.ty() == Type::I32 {
                return Some(left.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::I16(0))) = &*right.0 {
            if left.ty() == Type::I16 {
                return Some(left.clone());
            }
        }
        if let ExprKind::Const(Const(Literal::I8(0))) = &*right.0 {
            if left.ty() == Type::I8 {
                return Some(left.clone());
            }
        }
    }
    None
}

fn assoc_add(expr: &Expr) -> Option<Expr> {
    if let ExprKind::Binary {
        op: BinaryOp::Add,
        left,
        right,
    } = &*expr.0
    {
        if let ExprKind::Binary {
            op: inner_op,
            left: y_sub,
            right: z_sub,
        } = &*right.0
        {
            if *inner_op == BinaryOp::Add {
                let x = left.clone();
                let y = y_sub.clone();
                let z = z_sub.clone();
                let xy = Expr::binary(BinaryOp::Add, x, y);
                return Some(Expr::binary(BinaryOp::Add, xy, z));
            }
        }
    }
    None
}

fn fold_cmp_literals(expr: &Expr) -> Option<Expr> {
    if let ExprKind::Binary { op, left, right } = &*expr.0 {
        if op.is_comparison() {
            if let ExprKind::Const(Const(b_lit)) = &*left.0 {
                if let ExprKind::Const(Const(c_lit)) = &*right.0 {
                    if let Some(Literal::Bool(res_b)) = op.apply(b_lit, c_lit) {
                        return Some(Expr::bool_const(res_b));
                    }
                }
            }
        }
    }
    None
}

fn fold_select_and_mask(expr: &Expr) -> Option<Expr> {
    if let ExprKind::Select {
        cond,
        true_val,
        false_val,
    } = &*expr.0
    {
        // Match cond: (i & MASK) == 0u32
        if let ExprKind::Binary {
            op: BinaryOp::Eq,
            left: and_expr,
            right: right_zero,
        } = &*cond.0
        {
            // Ensure right side is the constant 0u32
            if let ExprKind::Const(Const(Literal::U32(0))) = &*right_zero.0 {
                // Match the AND: i & MASK
                if let ExprKind::Binary {
                    op: BinaryOp::And,
                    left: var_i,
                    right: mask_const,
                } = &*and_expr.0
                {
                    if let ExprKind::Const(Const(Literal::U32(mask))) = &*mask_const.0 {
                        // Ensure true_val is 0u32 and false_val is MASK
                        if let ExprKind::Const(Const(Literal::U32(0))) = &*true_val.0 {
                            if let ExprKind::Const(Const(Literal::U32(m2))) = &*false_val.0 {
                                if mask == m2 {
                                    // Rewrite to (i & MASK)
                                    return Some(Expr::binary(
                                        BinaryOp::And,
                                        var_i.clone(),
                                        Expr::u32_const(*mask),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

fn fold_and_mask_xor_chain(expr: &Expr) -> Option<Expr> {
    // Try to flatten a chain of XORs whose leaves are all (VAR & CONST)
    // Returns Some(VAR & combined_mask) if successful.

    // This will hold the common VAR once we see the first leaf.
    let mut var_expr: Option<Expr> = None;
    let mut masks: Vec<u32> = Vec::new();
    let mut ok = true;

    // Recursive helper: traverse `e`, collect masks if pattern matches.
    fn collect(e: &Expr, var_expr: &mut Option<Expr>, masks: &mut Vec<u32>, ok: &mut bool) {
        if !*ok {
            return;
        }
        match &*e.0 {
            ExprKind::Binary {
                op: BinaryOp::Xor,
                left,
                right,
            } => {
                collect(left, var_expr, masks, ok);
                collect(right, var_expr, masks, ok);
            }
            ExprKind::Binary {
                op: BinaryOp::And,
                left,
                right,
            } => {
                // Check for (VAR & CONST) or (CONST & VAR)
                match (&*left.0, &*right.0) {
                    (ExprKind::Var(_), ExprKind::Const(Const(Literal::U32(m)))) => {
                        // left is VAR, right is CONST(m)
                        let this_var = left.clone();
                        if let Some(v) = var_expr {
                            if *v != this_var {
                                *ok = false;
                                return;
                            }
                        } else {
                            *var_expr = Some(this_var.clone());
                        }
                        masks.push(*m);
                    }
                    (ExprKind::Const(Const(Literal::U32(m))), ExprKind::Var(_)) => {
                        // reversed: CONST(m) & VAR
                        let this_var = right.clone();
                        if let Some(v) = var_expr {
                            if *v != this_var {
                                *ok = false;
                                return;
                            }
                        } else {
                            *var_expr = Some(this_var.clone());
                        }
                        masks.push(*m);
                    }
                    _ => {
                        *ok = false;
                    }
                }
            }
            _ => {
                *ok = false;
            }
        }
    }

    collect(expr, &mut var_expr, &mut masks, &mut ok);

    if !ok {
        return None;
    }
    let var = var_expr?;
    if masks.is_empty() {
        return None;
    }
    // Compute cumulative XOR of all mask constants
    let combined = masks.into_iter().fold(0u32, |acc, m| acc ^ m);
    Some(Expr::binary(
        BinaryOp::And,
        var.clone(),
        Expr::u32_const(combined),
    ))
}

/// =============================================
/// 4) Const Array of Rules
/// =============================================

pub const RULES: &[RewriteRule] = &[
    fold_bin_literals,
    fold_add_zero_left,
    fold_add_zero_right,
    fold_xor_zero_left,
    fold_xor_zero_right,
    fold_cmp_literals,
    assoc_add,
    fold_select_and_mask,
    fold_and_mask_xor_chain,
];

/// =============================================
/// 5) Tests
/// =============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_literal_helpers_display() {
        assert_eq!(format!("{}", Expr::i8_const(-5)), "-5i8");
        assert_eq!(format!("{}", Expr::u8_const(10)), "10u8");
        assert_eq!(format!("{}", Expr::i16_const(-200)), "-200i16");
        assert_eq!(format!("{}", Expr::u16_const(500)), "500u16");
        assert_eq!(format!("{}", Expr::i32_const(-12345)), "-12345i32");
        assert_eq!(format!("{}", Expr::u32_const(12345)), "12345u32");
        assert_eq!(format!("{}", Expr::f32_const(1.5)), "1.5f32");
        assert_eq!(format!("{}", Expr::f64_const(2.75)), "2.75f64");
        assert_eq!(format!("{}", Expr::bool_const(true)), "true");
        assert_eq!(format!("{}", Expr::bool_const(false)), "false");
    }

    #[test]
    fn test_fold_bin_literals() {
        let three = Expr::i32_const(3);
        let four = Expr::i32_const(4);
        let add = Expr::binary(BinaryOp::Add, three.clone(), four.clone());
        let result = rewrite_fixpoint(add, RULES);
        assert_eq!(format!("{}", result), "7i32");
    }

    #[test]
    fn test_fold_add_zero_left() {
        let a = Expr::var("a", Type::I32);
        let zero = Expr::i32_const(0);
        let add = Expr::binary(BinaryOp::Add, zero.clone(), a.clone());
        let result = rewrite_fixpoint(add, RULES);
        assert_eq!(format!("{}", result), "a");
    }

    #[test]
    fn test_fold_add_zero_right() {
        let a = Expr::var("a", Type::I32);
        let zero = Expr::i32_const(0);
        let add = Expr::binary(BinaryOp::Add, a.clone(), zero.clone());
        let result = rewrite_fixpoint(add, RULES);
        assert_eq!(format!("{}", result), "a");
    }

    #[test]
    fn test_fold_xor_zero_left() {
        let x = Expr::var("x", Type::U32);
        let zero = Expr::u32_const(0);
        let xor = Expr::binary(BinaryOp::Xor, zero.clone(), x.clone());
        let result = rewrite_fixpoint(xor, RULES);
        assert_eq!(format!("{}", result), "x");
    }

    #[test]
    fn test_fold_xor_zero_right() {
        let x = Expr::var("x", Type::U32);
        let zero = Expr::u32_const(0);
        let xor = Expr::binary(BinaryOp::Xor, x.clone(), zero.clone());
        let result = rewrite_fixpoint(xor, RULES);
        assert_eq!(format!("{}", result), "x");
    }

    #[test]
    fn test_assoc_add() {
        let a = Expr::var("a", Type::I32);
        let b = Expr::var("b", Type::I32);
        let c = Expr::var("c", Type::I32);
        // (a + (b + c))
        let inner = Expr::binary(BinaryOp::Add, b.clone(), c.clone());
        let top = Expr::binary(BinaryOp::Add, a.clone(), inner);
        let result = rewrite_fixpoint(top, RULES);
        assert_eq!(format!("{}", result), "((a + b) + c)");
    }

    #[test]
    fn test_fold_cmp_literals_true() {
        let five = Expr::i32_const(5);
        let cmp = Expr::binary(BinaryOp::Eq, five.clone(), five.clone());
        let result = rewrite_fixpoint(cmp, RULES);
        assert_eq!(format!("{}", result), "true");
    }

    #[test]
    fn test_combined_rewrites() {
        // ((a + 0i32) + (3i32 + 4i32)) → (a + 7i32)
        let a = Expr::var("a", Type::I32);
        let zero = Expr::i32_const(0);
        let three = Expr::i32_const(3);
        let four = Expr::i32_const(4);

        let left1 = Expr::binary(BinaryOp::Add, a.clone(), zero.clone());
        let right = Expr::binary(BinaryOp::Add, three.clone(), four.clone());
        let top = Expr::binary(BinaryOp::Add, left1, right);

        let result = rewrite_fixpoint(top, RULES);
        assert_eq!(format!("{}", result), "(a + 7i32)");
    }

    #[test]
    fn test_fold_xor_chain_with_initial_zero() {
        // Original expression:
        //   ((((0u32 ^ sel1) ^ sel2) ^ sel3) ^ sel4) ^ sel5
        //
        // After rewriting, initial `0u32 ^ sel1` should fold to sel1, leaving
        // ((((sel1 ^ sel2) ^ sel3) ^ sel4) ^ sel5).

        let i = Expr::var("i", Type::U32);
        let one = Expr::u32_const(1);
        let two = Expr::u32_const(2);
        let four = Expr::u32_const(4);
        let eight = Expr::u32_const(8);
        let sixteen = Expr::u32_const(16);
        let zero_u32 = Expr::u32_const(0);

        // sel1 = select((i & 1u32) == 0u32, 0u32, 1u32)
        let sel1 = Expr::select(
            Expr::binary(
                BinaryOp::Eq,
                Expr::binary(BinaryOp::And, i.clone(), one.clone()),
                Expr::u32_const(0),
            ),
            zero_u32.clone(),
            one.clone(),
        );
        // sel2 = select((i & 2u32) == 0u32, 0u32, 2u32)
        let sel2 = Expr::select(
            Expr::binary(
                BinaryOp::Eq,
                Expr::binary(BinaryOp::And, i.clone(), two.clone()),
                Expr::u32_const(0),
            ),
            zero_u32.clone(),
            two.clone(),
        );
        // sel3 = select((i & 4u32) == 0u32, 0u32, 4u32)
        let sel3 = Expr::select(
            Expr::binary(
                BinaryOp::Eq,
                Expr::binary(BinaryOp::And, i.clone(), four.clone()),
                Expr::u32_const(0),
            ),
            zero_u32.clone(),
            four.clone(),
        );
        // sel4 = select((i & 8u32) == 0u32, 0u32, 8u32)
        let sel4 = Expr::select(
            Expr::binary(
                BinaryOp::Eq,
                Expr::binary(BinaryOp::And, i.clone(), eight.clone()),
                Expr::u32_const(0),
            ),
            zero_u32.clone(),
            eight.clone(),
        );
        // sel5 = select((i & 16u32) == 0u32, 0u32, 16u32)
        let sel5 = Expr::select(
            Expr::binary(
                BinaryOp::Eq,
                Expr::binary(BinaryOp::And, i.clone(), sixteen.clone()),
                Expr::u32_const(0),
            ),
            zero_u32.clone(),
            sixteen.clone(),
        );

        // Build (((((0u32 ^ sel1) ^ sel2) ^ sel3) ^ sel4) ^ sel5)
        let expr = Expr::binary(
            BinaryOp::Xor,
            Expr::binary(
                BinaryOp::Xor,
                Expr::binary(
                    BinaryOp::Xor,
                    Expr::binary(
                        BinaryOp::Xor,
                        Expr::binary(BinaryOp::Xor, zero_u32.clone(), sel1.clone()),
                        sel2.clone(),
                    ),
                    sel3.clone(),
                ),
                sel4.clone(),
            ),
            sel5.clone(),
        );

        let rewritten = rewrite_fixpoint(
            expr.clone(),
            &[
                fold_bin_literals,
                fold_add_zero_left,
                fold_add_zero_right,
                fold_xor_zero_left,
                fold_xor_zero_right,
                fold_cmp_literals,
                assoc_add,
            ],
        );

        // Build the expected expression directly: ((((sel1 ^ sel2) ^ sel3) ^ sel4) ^ sel5)
        let expected = Expr::binary(
            BinaryOp::Xor,
            Expr::binary(
                BinaryOp::Xor,
                Expr::binary(
                    BinaryOp::Xor,
                    Expr::binary(BinaryOp::Xor, sel1.clone(), sel2.clone()),
                    sel3.clone(),
                ),
                sel4.clone(),
            ),
            sel5.clone(),
        );

        assert_eq!(rewritten, expected);
    }

    #[test]
    fn test_fold_select_and_mask() {
        // select(((i & 2u32) == 0u32), 0u32, 2u32)  ==>  (i & 2u32)
        let i = Expr::var("i", Type::U32);
        let mask = Expr::u32_const(2);
        let zero = Expr::u32_const(0);

        let cond = Expr::binary(
            BinaryOp::Eq,
            Expr::binary(BinaryOp::And, i.clone(), mask.clone()),
            zero.clone(),
        );
        let sel = Expr::select(cond, zero.clone(), mask.clone());

        let rewritten = rewrite_once(&sel, &[fold_select_and_mask]);
        assert_eq!(format!("{}", rewritten), "(i & 2u32)");
    }

    #[test]
    fn test_fold_and_mask_xor_chain() {
        // ((((i & 1u32) ^ (i & 2u32)) ^ (i & 4u32)) ^ (i & 8u32)) ^ (i & 16u32)
        // should fold to (i & (1^2^4^8^16)) = (i & 31u32)

        let i = Expr::var("i", Type::U32);
        let one = Expr::u32_const(1);
        let two = Expr::u32_const(2);
        let four = Expr::u32_const(4);
        let eight = Expr::u32_const(8);
        let sixteen = Expr::u32_const(16);

        let expr = Expr::binary(
            BinaryOp::Xor,
            Expr::binary(
                BinaryOp::Xor,
                Expr::binary(
                    BinaryOp::Xor,
                    Expr::binary(
                        BinaryOp::Xor,
                        Expr::binary(BinaryOp::And, i.clone(), one.clone()),
                        Expr::binary(BinaryOp::And, i.clone(), two.clone()),
                    ),
                    Expr::binary(BinaryOp::And, i.clone(), four.clone()),
                ),
                Expr::binary(BinaryOp::And, i.clone(), eight.clone()),
            ),
            Expr::binary(BinaryOp::And, i.clone(), sixteen.clone()),
        );

        let rewritten = rewrite_once(&expr, &[fold_and_mask_xor_chain]);
        assert_eq!(format!("{}", rewritten), "(i & 31u32)");
    }
}
