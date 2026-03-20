#![cfg(target_family = "wasm")]
#![allow(non_snake_case)]

use fragile::Fragile;
use js_sys::{Array, Function, Object};
use minijinja::machinery::{ast::*, parse, WhitespaceConfig};
use minijinja::syntax::SyntaxConfig;
use minijinja::{self as mj, Error, ErrorKind, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use wasm_bindgen::prelude::*;

/// Represents a MiniJinja environment.
#[wasm_bindgen]
#[derive(Clone)]
pub struct Environment {
    inner: mj::Environment<'static>,
}

#[wasm_bindgen]
impl Environment {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let inner = mj::Environment::new();
        Self { inner }
    }

    /// Registers a new template by name and source.
    pub fn addTemplate(&mut self, name: &str, source: &str) -> Result<(), JsError> {
        self.inner
            .add_template_owned(name.to_string(), source.to_string())
            .map_err(convert_error)
    }

    /// Removes a template by name.
    pub fn removeTemplate(&mut self, name: &str) {
        self.inner.remove_template(name);
    }

    /// Clears all templates from the environment.
    pub fn clearTemplates(&mut self) {
        self.inner.clear_templates();
    }

    /// Renders a registered template by name with the given context.
    pub fn renderTemplate(&mut self, name: &str, ctx: JsValue) -> Result<String, JsError> {
        let ctx = js_to_mj_value(ctx)?;
        let t = self.inner.get_template(name).map_err(convert_error)?;
        t.render(ctx).map_err(convert_error)
    }

    /// Renders a string template with the given context.
    ///
    /// This is useful for one-off template rendering without registering the template.  The
    /// template is parsed and rendered immediately.
    pub fn renderStr(&mut self, source: &str, ctx: JsValue) -> Result<String, JsError> {
        let ctx = js_to_mj_value(ctx)?;
        self.inner.render_str(source, ctx).map_err(convert_error)
    }

    /// Like `renderStr` but with a named template for auto escape detection.
    pub fn renderNamedStr(
        &mut self,
        name: &str,
        source: &str,
        ctx: JsValue,
    ) -> Result<String, JsError> {
        let ctx = js_to_mj_value(ctx)?;
        self.inner
            .render_named_str(name, source, ctx)
            .map_err(convert_error)
    }

    /// Evaluates an expression with the given context.
    ///
    /// This is useful for evaluating expressions outside of templates.  The expression is
    /// parsed and evaluated immediately.
    pub fn evalExpr(&mut self, expr: &str, ctx: JsValue) -> Result<JsValue, JsError> {
        let ctx = js_to_mj_value(ctx)?;
        let e = self.inner.compile_expression(expr).map_err(convert_error)?;
        let result = e.eval(ctx).map_err(convert_error)?;
        serde_wasm_bindgen::to_value(&result).map_err(|err| JsError::new(&err.to_string()))
    }

    /// Registers a filter function.
    pub fn addFilter(&mut self, name: &str, func: Function) {
        self.inner
            .add_filter(name.to_string(), create_js_callback(func));
    }

    /// Registers a test function.
    pub fn addTest(&mut self, name: &str, func: Function) {
        self.inner
            .add_test(name.to_string(), create_js_callback(func));
    }

    pub fn findVars(&self, source: &str) -> Vec<String> {
        let ast = parse(
            source,
            "<template>",
            SyntaxConfig::default(),
            WhitespaceConfig::default(),
        )
        .unwrap();

        let mut undeclared = BTreeSet::new();
        let mut scope: VecDeque<BTreeSet<String>> = VecDeque::new();

        scope.push_back(BTreeSet::new());

        fn is_declared(scope: &VecDeque<BTreeSet<String>>, name: &str) -> bool {
            scope.iter().rev().any(|s| s.contains(name))
        }

        fn declare(scope: &mut VecDeque<BTreeSet<String>>, name: String) {
            if let Some(current) = scope.back_mut() {
                current.insert(name);
            }
        }

        fn walk_call_args(
            args: &[CallArg],
            scope: &VecDeque<BTreeSet<String>>,
            undeclared: &mut BTreeSet<String>,
        ) {
            for arg in args {
                match arg {
                    CallArg::Pos(e) | CallArg::PosSplat(e) | CallArg::KwargSplat(e) => {
                        walk_expr(e, scope, undeclared)
                    }

                    CallArg::Kwarg(_, e) => walk_expr(e, scope, undeclared),
                }
            }
        }

        fn walk_stmt(
            stmt: &Stmt,
            scope: &mut VecDeque<BTreeSet<String>>,
            undeclared: &mut BTreeSet<String>,
        ) {
            match stmt {
                Stmt::Template(tpl) => {
                    for s in &tpl.children {
                        walk_stmt(s, scope, undeclared);
                    }
                }

                Stmt::EmitExpr(tpl) => walk_expr(&tpl.expr, scope, undeclared),

                Stmt::ForLoop(fl) => {
                    walk_expr(&fl.iter, scope, undeclared);

                    scope.push_back(BTreeSet::new());
                    collect_target_names(&fl.target, scope);

                    for s in &fl.body {
                        walk_stmt(s, scope, undeclared);
                    }

                    scope.pop_back();
                }

                Stmt::IfCond(ic) => {
                    walk_expr(&ic.expr, scope, undeclared);

                    scope.push_back(BTreeSet::new());
                    for s in &ic.true_body {
                        walk_stmt(s, scope, undeclared);
                    }
                    scope.pop_back();

                    scope.push_back(BTreeSet::new());
                    for s in &ic.false_body {
                        walk_stmt(s, scope, undeclared);
                    }
                    scope.pop_back();
                }

                Stmt::Set(s) => {
                    walk_expr(&s.expr, scope, undeclared);
                    collect_target_names(&s.target, scope);
                }

                _ => {}
            }
        }

        fn walk_expr(
            expr: &Expr,
            scope: &VecDeque<BTreeSet<String>>,
            undeclared: &mut BTreeSet<String>,
        ) {
            match expr {
                Expr::Var(v) => {
                    if !is_declared(scope, v.id) {
                        undeclared.insert(v.id.to_string());
                    }
                }

                Expr::GetAttr(obj) => {
                    walk_expr(&obj.expr, scope, undeclared);
                }

                Expr::GetItem(obj) => {
                    walk_expr(&obj.expr, scope, undeclared);
                    walk_expr(&obj.subscript_expr, scope, undeclared);
                }

                Expr::Call(func) => {
                    walk_expr(&func.expr, scope, undeclared);
                    walk_call_args(&func.args, scope, undeclared);
                }

                Expr::BinOp(op) => {
                    walk_expr(&op.left, scope, undeclared);
                    walk_expr(&op.right, scope, undeclared);
                }

                Expr::UnaryOp(op) => {
                    walk_expr(&op.expr, scope, undeclared);
                }

                Expr::Filter(filter) => {
                    if let Some(expr) = &filter.expr {
                        walk_expr(expr, scope, undeclared);
                    }
                    walk_call_args(&filter.args, scope, undeclared);
                }

                Expr::Test(test) => {
                    walk_expr(&test.expr, scope, undeclared);
                    walk_call_args(&test.args, scope, undeclared);
                }

                _ => {}
            }
        }

        fn collect_target_names(target: &Expr, scope: &mut VecDeque<BTreeSet<String>>) {
            match target {
                Expr::Var(v) => declare(scope, v.id.to_string()),
                _ => {}
            }
        }

        walk_stmt(&ast, &mut scope, &mut undeclared);

        undeclared.into_iter().collect()
    }

    /// Enables or disables block trimming.
    #[wasm_bindgen(getter)]
    pub fn trimBlocks(&self) -> bool {
        self.inner.trim_blocks()
    }

    #[wasm_bindgen(setter)]
    pub fn set_trimBlocks(&mut self, yes: bool) {
        self.inner.set_trim_blocks(yes);
    }

    /// Enables or disables the lstrip blocks feature.
    #[wasm_bindgen(getter)]
    pub fn lstripBlocks(&self) -> bool {
        self.inner.lstrip_blocks()
    }

    #[wasm_bindgen(setter)]
    pub fn set_lstripBlocks(&mut self, yes: bool) {
        self.inner.set_lstrip_blocks(yes);
    }

    /// Enables or disables keeping of the final newline.
    #[wasm_bindgen(getter)]
    pub fn keepTrailingNewline(&self) -> bool {
        self.inner.keep_trailing_newline()
    }

    #[wasm_bindgen(setter)]
    pub fn set_keepTrailingNewline(&mut self, yes: bool) {
        self.inner.set_keep_trailing_newline(yes);
    }

    /// Reconfigures the behavior of undefined variables.
    #[wasm_bindgen(getter)]
    pub fn undefinedBehavior(&self) -> UndefinedBehavior {
        self.inner.undefined_behavior().into()
    }

    #[wasm_bindgen(setter)]
    pub fn set_undefinedBehavior(&mut self, value: UndefinedBehavior) -> Result<(), JsError> {
        self.inner.set_undefined_behavior(value.into());
        Ok(())
    }

    /// Registers a value as global.
    #[wasm_bindgen]
    pub fn addGlobal(&mut self, name: &str, value: JsValue) -> Result<(), JsError> {
        self.inner
            .add_global(name.to_string(), js_to_mj_value(value)?);
        Ok(())
    }

    /// Removes a global again.
    #[wasm_bindgen]
    pub fn removeGlobal(&mut self, name: &str) {
        self.inner.remove_global(name);
    }
}

#[wasm_bindgen]
#[derive(Copy, Clone, Debug)]
pub enum UndefinedBehavior {
    Strict = "strict",
    Chainable = "chainable",
    Lenient = "lenient",
    SemiStrict = "semi_strct",
}

impl From<mj::UndefinedBehavior> for UndefinedBehavior {
    fn from(value: mj::UndefinedBehavior) -> Self {
        match value {
            mj::UndefinedBehavior::Strict => UndefinedBehavior::Strict,
            mj::UndefinedBehavior::Chainable => UndefinedBehavior::Chainable,
            mj::UndefinedBehavior::Lenient => UndefinedBehavior::Lenient,
            mj::UndefinedBehavior::SemiStrict => UndefinedBehavior::SemiStrict,
            _ => unreachable!(),
        }
    }
}

impl From<UndefinedBehavior> for mj::UndefinedBehavior {
    fn from(value: UndefinedBehavior) -> Self {
        match value {
            UndefinedBehavior::Strict => mj::UndefinedBehavior::Strict,
            UndefinedBehavior::Chainable => mj::UndefinedBehavior::Chainable,
            UndefinedBehavior::Lenient => mj::UndefinedBehavior::Lenient,
            UndefinedBehavior::SemiStrict => mj::UndefinedBehavior::SemiStrict,
            _ => unreachable!(),
        }
    }
}

fn convert_error(err: minijinja::Error) -> JsError {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    JsError::new(&format!("{:#}", err))
}

fn js_to_mj_value(value: JsValue) -> Result<Value, JsError> {
    if value.is_function() {
        Ok(Value::from_function(create_js_callback(Function::from(
            value,
        ))))
    } else if value.is_array() {
        let arr = Array::from(&value);
        let mut rv = Vec::new();
        for i in 0..arr.length() {
            rv.push(js_to_mj_value(arr.get(i))?);
        }
        Ok(Value::from(rv))
    } else if value.is_object() {
        let obj = Object::from(value);
        let entries = Object::entries(&obj);
        let mut map = BTreeMap::new();
        for i in 0..entries.length() {
            let entry = Array::from(&entries.get(i));
            let key = entry.get(0);
            let value = entry.get(1);
            map.insert(js_to_mj_value(key)?, js_to_mj_value(value)?);
        }
        Ok(Value::from(map))
    } else if let Some(s) = value.as_string() {
        Ok(Value::from(s))
    } else if let Some(n) = value.as_f64() {
        Ok(Value::from(n))
    } else if let Some(b) = value.as_bool() {
        Ok(Value::from(b))
    } else if value.is_null() || value.is_undefined() {
        Ok(Value::from(()))
    } else {
        Err(JsError::new("unsupported value type"))
    }
}

fn create_js_callback(func: Function) -> impl Fn(&[Value]) -> Result<Value, Error> {
    let fragile_func = Fragile::new(func);
    move |args: &[Value]| -> Result<Value, Error> {
        let values = js_sys::Array::new();
        for arg in args {
            values.push(&serde_wasm_bindgen::to_value(arg).map_err(|err| {
                Error::new(
                    ErrorKind::InvalidOperation,
                    format!("failed to convert argument: {}", err),
                )
            })?);
        }
        let func = fragile_func.get();
        let rv = func.apply(&JsValue::null(), &values).unwrap();
        let ctx: Value = js_to_mj_value(rv).map_err(|err| {
            Error::new(
                ErrorKind::InvalidOperation,
                format!("failed to convert result: {:?}", err),
            )
        })?;
        Ok(ctx)
    }
}
