// RUN: %clang_cc1 -fsyntax-only -verify %s

/***********************************
 *  No Thread Safety Analysis (noanal)
 ***********************************/

// FIXME: Right now we cannot parse attributes put on function definitions
// We would like to patch this at some point.

#if !__has_attribute(no_thread_safety_analysis)
#error "Should support no_thread_safety_analysis attribute"
#endif

void noanal_function() __attribute__((no_thread_safety_analysis));

void noanal_function() __attribute__((no_thread_safety_analysis(1))); // \
  expected-error {{attribute takes no arguments}}

int noanal_testfn(int y) __attribute__((no_thread_safety_analysis));

int noanal_testfn(int y) {
  int x __attribute__((no_thread_safety_analysis)) = y; // \
    expected-warning {{'no_thread_safety_analysis' attribute only applies to functions and methods}}
  return x;
};

int noanal_test_var __attribute__((no_thread_safety_analysis)); // \
  expected-warning {{'no_thread_safety_analysis' attribute only applies to functions and methods}}

class NoanalFoo {
 private:
  int test_field __attribute__((no_thread_safety_analysis)); // \
    expected-warning {{'no_thread_safety_analysis' attribute only applies to functions and methods}}
  void test_method() __attribute__((no_thread_safety_analysis));
};

class __attribute__((no_thread_safety_analysis)) NoanalTestClass { // \
    expected-warning {{'no_thread_safety_analysis' attribute only applies to functions and methods}}
};

void noanal_fun_params(int lvar __attribute__((no_thread_safety_analysis))); // \
  expected-warning {{'no_thread_safety_analysis' attribute only applies to functions and methods}}


/***********************************
 *  Guarded Var Attribute (gv)
 ***********************************/

#if !__has_attribute(guarded_var)
#error "Should support guarded_var attribute"
#endif

int gv_var_noargs __attribute__((guarded_var));

int gv_var_args __attribute__((guarded_var(1))); // \
    expected-error {{attribute takes no arguments}}

class GVFoo {
 private:
  int gv_field_noargs __attribute__((guarded_var));
  int gv_field_args __attribute__((guarded_var(1))); // \
      expected-error {{attribute takes no arguments}}
};

class __attribute__((guarded_var)) GV { // \
      expected-warning {{'guarded_var' attribute only applies to fields and global variables}}
};

void gv_function() __attribute__((guarded_var)); // \
    expected-warning {{'guarded_var' attribute only applies to fields and global variables}}

void gv_function_params(int gv_lvar __attribute__((guarded_var))); // \
    expected-warning {{'guarded_var' attribute only applies to fields and global variables}}

int gv_testfn(int y){
  int x __attribute__((guarded_var)) = y; // \
      expected-warning {{'guarded_var' attribute only applies to fields and global variables}}
  return x;
}

/***********************************
 *  Pt Guarded Var Attribute (pgv)
 ***********************************/

//FIXME: add support for boost::scoped_ptr<int> fancyptr  and references

#if !__has_attribute(pt_guarded_var)
#error "Should support pt_guarded_var attribute"
#endif

int *pgv_pt_var_noargs __attribute__((pt_guarded_var));

int pgv_var_noargs __attribute__((pt_guarded_var)); // \
    expected-warning {{'pt_guarded_var' only applies to pointer types; type here is 'int'}}

class PGVFoo {
 private:
  int *pt_field_noargs __attribute__((pt_guarded_var));
  int field_noargs __attribute__((pt_guarded_var)); // \
    expected-warning {{'pt_guarded_var' only applies to pointer types; type here is 'int'}}
  int *gv_field_args __attribute__((pt_guarded_var(1))); // \
    expected-error {{attribute takes no arguments}}
};

class __attribute__((pt_guarded_var)) PGV { // \
    expected-warning {{'pt_guarded_var' attribute only applies to fields and global variables}}
};

int *pgv_var_args __attribute__((pt_guarded_var(1))); // \
  expected-error {{attribute takes no arguments}}


void pgv_function() __attribute__((pt_guarded_var)); // \
  expected-warning {{'pt_guarded_var' attribute only applies to fields and global variables}}

void pgv_function_params(int *gv_lvar __attribute__((pt_guarded_var))); // \
  expected-warning {{'pt_guarded_var' attribute only applies to fields and global variables}}

void pgv_testfn(int y){
  int *x __attribute__((pt_guarded_var)) = new int(0); // \
    expected-warning {{'pt_guarded_var' attribute only applies to fields and global variables}}
  delete x;
}

/***********************************
 *  Lockable Attribute (l)
 ***********************************/

//FIXME: In future we may want to add support for structs, ObjC classes, etc.

#if !__has_attribute(lockable)
#error "Should support lockable attribute"
#endif

class __attribute__((lockable)) LTestClass {
};

class __attribute__((lockable (1))) LTestClass_args { // \
    expected-error {{attribute takes no arguments}}
};

void l_test_function() __attribute__((lockable));  // \
  expected-warning {{'lockable' attribute only applies to classes}}

int l_testfn(int y) {
  int x __attribute__((lockable)) = y; // \
    expected-warning {{'lockable' attribute only applies to classes}}
  return x;
}

int l_test_var __attribute__((lockable)); // \
  expected-warning {{'lockable' attribute only applies to classes}}

class LFoo {
 private:
  int test_field __attribute__((lockable)); // \
    expected-warning {{'lockable' attribute only applies to classes}}
  void test_method() __attribute__((lockable)); // \
    expected-warning {{'lockable' attribute only applies to classes}}
};


void l_function_params(int lvar __attribute__((lockable))); // \
  expected-warning {{'lockable' attribute only applies to classes}}


/***********************************
 *  Scoped Lockable Attribute (sl)
 ***********************************/

#if !__has_attribute(scoped_lockable)
#error "Should support scoped_lockable attribute"
#endif

class __attribute__((scoped_lockable)) SLTestClass {
};

class __attribute__((scoped_lockable (1))) SLTestClass_args { // \
    expected-error {{attribute takes no arguments}}
};

void sl_test_function() __attribute__((scoped_lockable));  // \
  expected-warning {{'scoped_lockable' attribute only applies to classes}}

int sl_testfn(int y) {
  int x __attribute__((scoped_lockable)) = y; // \
    expected-warning {{'scoped_lockable' attribute only applies to classes}}
  return x;
}

int sl_test_var __attribute__((scoped_lockable)); // \
  expected-warning {{'scoped_lockable' attribute only applies to classes}}

class SLFoo {
 private:
  int test_field __attribute__((scoped_lockable)); // \
    expected-warning {{'scoped_lockable' attribute only applies to classes}}
  void test_method() __attribute__((scoped_lockable)); // \
    expected-warning {{'scoped_lockable' attribute only applies to classes}}
};


void sl_function_params(int lvar __attribute__((scoped_lockable))); // \
  expected-warning {{'scoped_lockable' attribute only applies to classes}}
