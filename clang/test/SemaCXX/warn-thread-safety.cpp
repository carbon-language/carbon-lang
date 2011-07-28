// RUN: %clang_cc1 -fsyntax-only -verify %s


/**
 *  Helper fields
 */

class __attribute__((lockable)) Mu {
};

Mu mu1;
Mu mu2;

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


/***********************************
 *  Guarded By Attribute (gb)
 ***********************************/

// FIXME: Would we like this attribute to take more than 1 arg?

#if !__has_attribute(guarded_by)
#error "Should support guarded_by attribute"
#endif

//1. Check applied to the right types & argument number

int gb_var_arg __attribute__((guarded_by(mu1)));

int gb_var_args __attribute__((guarded_by(mu1, mu2))); // \
  expected-error {{attribute takes one argument}}

int gb_var_noargs __attribute__((guarded_by)); // \
  expected-error {{attribute takes one argument}}

class GBFoo {
 private:
  int gb_field_noargs __attribute__((guarded_by)); // \
    expected-error {{attribute takes one argument}}
  int gb_field_args __attribute__((guarded_by(mu1)));
};

class __attribute__((guarded_by(mu1))) GB { // \
      expected-warning {{'guarded_by' attribute only applies to fields and global variables}}
};

void gb_function() __attribute__((guarded_by(mu1))); // \
    expected-warning {{'guarded_by' attribute only applies to fields and global variables}}

void gb_function_params(int gv_lvar __attribute__((guarded_by(mu1)))); // \
    expected-warning {{'guarded_by' attribute only applies to fields and global variables}}

int gb_testfn(int y){
  int x __attribute__((guarded_by(mu1))) = y; // \
      expected-warning {{'guarded_by' attribute only applies to fields and global variables}}
  return x;
}

//2.Deal with argument parsing:
// grab token stream parsing from C++0x branch
// possibly create new, more permissive category for gcc attributes

//foo
//foo.bar
//foo.bar->baz
//foo.bar()->baz()->a
//&foo
//*foo

//3.
// Thread Safety analysis tests


/***********************************
 *  Pt Guarded By Attribute (pgb)
 ***********************************/

#if !__has_attribute(pt_guarded_by)
#error "Should support pt_guarded_by attribute"
#endif

//1. Check applied to the right types & argument number

int *pgb_var_noargs __attribute__((pt_guarded_by)); // \
  expected-error {{attribute takes one argument}}

int *pgb_ptr_var_arg __attribute__((pt_guarded_by(mu1)));

int *pgb_ptr_var_args __attribute__((guarded_by(mu1, mu2))); // \
  expected-error {{attribute takes one argument}}

int pgb_var_args __attribute__((pt_guarded_by(mu1))); // \
    expected-warning {{'pt_guarded_by' only applies to pointer types; type here is 'int'}}

class PGBFoo {
 private:
  int *pgb_field_noargs __attribute__((pt_guarded_by)); // \
    expected-error {{attribute takes one argument}}
  int *pgb_field_args __attribute__((pt_guarded_by(mu1)));
};

class __attribute__((pt_guarded_by(mu1))) PGB { // \
      expected-warning {{'pt_guarded_by' attribute only applies to fields and global variables}}
};

void pgb_function() __attribute__((pt_guarded_by(mu1))); // \
    expected-warning {{'pt_guarded_by' attribute only applies to fields and global variables}}

void pgb_function_params(int gv_lvar __attribute__((pt_guarded_by(mu1)))); // \
    expected-warning {{'pt_guarded_by' attribute only applies to fields and global variables}}

void pgb_testfn(int y){
  int *x __attribute__((pt_guarded_by(mu1))) = new int(0); // \
      expected-warning {{'pt_guarded_by' attribute only applies to fields and global variables}}
  delete x;
}

/***********************************
 *  Acquired After (aa)
 ***********************************/

// FIXME: Would we like this attribute to take more than 1 arg?
// FIXME: What about pointers to locks?

#if !__has_attribute(acquired_after)
#error "Should support acquired_after attribute"
#endif

Mu mu_aa __attribute__((acquired_after(mu1)));

Mu aa_var_noargs __attribute__((acquired_after)); // \
  expected-error {{attribute takes at least 1 argument}}

class AAFoo {
 private:
  Mu aa_field_noargs __attribute__((acquired_after)); // \
    expected-error {{attribute takes at least 1 argument}}
  Mu aa_field_args __attribute__((acquired_after(mu1)));
};

class __attribute__((acquired_after(mu1))) AA { // \
      expected-warning {{'acquired_after' attribute only applies to fields and global variables}}
};

void aa_function() __attribute__((acquired_after(mu1))); // \
    expected-warning {{'acquired_after' attribute only applies to fields and global variables}}

void aa_function_params(int gv_lvar __attribute__((acquired_after(mu1)))); // \
    expected-warning {{'acquired_after' attribute only applies to fields and global variables}}

void aa_testfn(int y){
  Mu x __attribute__((acquired_after(mu1))) = Mu(); // \
      expected-warning {{'acquired_after' attribute only applies to fields and global variables}}
}

// Note: illegal int aa_int __attribute__((acquired_after(mu1))) will
// be taken care of by warnings that aa__int is not lockable.


/***********************************
 *  Acquired Before (ab)
 ***********************************/

#if !__has_attribute(acquired_before)
#error "Should support acquired_before attribute"
#endif

Mu mu_ab __attribute__((acquired_before(mu1)));

Mu ab_var_noargs __attribute__((acquired_before)); // \
  expected-error {{attribute takes at least 1 argument}}

class ABFoo {
 private:
  Mu ab_field_noargs __attribute__((acquired_before)); // \
    expected-error {{attribute takes at least 1 argument}}
  Mu ab_field_args __attribute__((acquired_before(mu1)));
};

class __attribute__((acquired_before(mu1))) AB { // \
      expected-warning {{'acquired_before' attribute only applies to fields and global variables}}
};

void ab_function() __attribute__((acquired_before(mu1))); // \
    expected-warning {{'acquired_before' attribute only applies to fields and global variables}}

void ab_function_params(int gv_lvar __attribute__((acquired_before(mu1)))); // \
    expected-warning {{'acquired_before' attribute only applies to fields and global variables}}

void ab_testfn(int y){
  Mu x __attribute__((acquired_before(mu1))) = Mu(); // \
      expected-warning {{'acquired_before' attribute only applies to fields and global variables}}
}

// Note: illegal int ab_int __attribute__((acquired_before(mu1))) will
// be taken care of by warnings that ab__int is not lockable.

/***********************************
 *  Exclusive Lock Function (elf)
 ***********************************/

#if !__has_attribute(exclusive_lock_function)
#error "Should support exclusive_lock_function attribute"
#endif

// takes zero or more arguments, all locks (vars/fields)

void elf_function() __attribute__((exclusive_lock_function));

void elf_function_args() __attribute__((exclusive_lock_function(mu1, mu2)));

int elf_testfn(int y) __attribute__((exclusive_lock_function));

int elf_testfn(int y) {
  int x __attribute__((exclusive_lock_function)) = y; // \
    expected-warning {{'exclusive_lock_function' attribute only applies to functions and methods}}
  return x;
};

int elf_test_var __attribute__((exclusive_lock_function)); // \
  expected-warning {{'exclusive_lock_function' attribute only applies to functions and methods}}

class ElfFoo {
 private:
  int test_field __attribute__((exclusive_lock_function)); // \
    expected-warning {{'exclusive_lock_function' attribute only applies to functions and methods}}
  void test_method() __attribute__((exclusive_lock_function));
};

class __attribute__((exclusive_lock_function)) ElfTestClass { // \
    expected-warning {{'exclusive_lock_function' attribute only applies to functions and methods}}
};

void elf_fun_params(int lvar __attribute__((exclusive_lock_function))); // \
  expected-warning {{'exclusive_lock_function' attribute only applies to functions and methods}}



/***********************************
 *  Shared Lock Function (slf)
 ***********************************/

#if !__has_attribute(shared_lock_function)
#error "Should support shared_lock_function attribute"
#endif

// takes zero or more arguments, all locks (vars/fields)

void slf_function() __attribute__((shared_lock_function));

void slf_function_args() __attribute__((shared_lock_function(mu1, mu2)));

int slf_testfn(int y) __attribute__((shared_lock_function));

int slf_testfn(int y) {
  int x __attribute__((shared_lock_function)) = y; // \
    expected-warning {{'shared_lock_function' attribute only applies to functions and methods}}
  return x;
};

int slf_test_var __attribute__((shared_lock_function)); // \
  expected-warning {{'shared_lock_function' attribute only applies to functions and methods}}

void slf_fun_params(int lvar __attribute__((shared_lock_function))); // \
  expected-warning {{'shared_lock_function' attribute only applies to functions and methods}}

class SlfFoo {
 private:
  int test_field __attribute__((shared_lock_function)); // \
    expected-warning {{'shared_lock_function' attribute only applies to functions and methods}}
  void test_method() __attribute__((shared_lock_function));
};

class __attribute__((shared_lock_function)) SlfTestClass { // \
    expected-warning {{'shared_lock_function' attribute only applies to functions and methods}}
};


/***********************************
 *  Exclusive TryLock Function (etf)
 ***********************************/

#if !__has_attribute(exclusive_trylock_function)
#error "Should support exclusive_trylock_function attribute"
#endif

// takes a mandatory boolean or integer argument specifying the retval
// plus an optional list of locks (vars/fields)

void etf_function() __attribute__((exclusive_trylock_function));  // \
  expected-error {{attribute takes attribute takes at least 1 argument arguments}}

void etf_function_args() __attribute__((exclusive_trylock_function(1, mu2)));

void etf_function_arg() __attribute__((exclusive_trylock_function(1)));

int etf_testfn(int y) __attribute__((exclusive_trylock_function(1)));

int etf_testfn(int y) {
  int x __attribute__((exclusive_trylock_function(1))) = y; // \
    expected-warning {{'exclusive_trylock_function' attribute only applies to functions and methods}}
  return x;
};

int etf_test_var __attribute__((exclusive_trylock_function(1))); // \
  expected-warning {{'exclusive_trylock_function' attribute only applies to functions and methods}}

class EtfFoo {
 private:
  int test_field __attribute__((exclusive_trylock_function(1))); // \
    expected-warning {{'exclusive_trylock_function' attribute only applies to functions and methods}}
  void test_method() __attribute__((exclusive_trylock_function(1)));
};

class __attribute__((exclusive_trylock_function(1))) EtfTestClass { // \
    expected-warning {{'exclusive_trylock_function' attribute only applies to functions and methods}}
};

void etf_fun_params(int lvar __attribute__((exclusive_trylock_function(1)))); // \
  expected-warning {{'exclusive_trylock_function' attribute only applies to functions and methods}}



/***********************************
 *  Shared TryLock Function (stf)
 ***********************************/

#if !__has_attribute(shared_trylock_function)
#error "Should support shared_trylock_function attribute"
#endif

// takes a mandatory boolean or integer argument specifying the retval
// plus an optional list of locks (vars/fields)

void stf_function() __attribute__((shared_trylock_function));  // \
  expected-error {{attribute takes at least 1 argument}}

void stf_function_args() __attribute__((shared_trylock_function(1, mu2)));

void stf_function_arg() __attribute__((shared_trylock_function(1)));

int stf_testfn(int y) __attribute__((shared_trylock_function(1)));

int stf_testfn(int y) {
  int x __attribute__((shared_trylock_function(1))) = y; // \
    expected-warning {{'shared_trylock_function' attribute only applies to functions and methods}}
  return x;
};

int stf_test_var __attribute__((shared_trylock_function(1))); // \
  expected-warning {{'shared_trylock_function' attribute only applies to functions and methods}}

void stf_fun_params(int lvar __attribute__((shared_trylock_function(1)))); // \
  expected-warning {{'shared_trylock_function' attribute only applies to functions and methods}}


class StfFoo {
 private:
  int test_field __attribute__((shared_trylock_function(1))); // \
    expected-warning {{'shared_trylock_function' attribute only applies to functions and methods}}
  void test_method() __attribute__((shared_trylock_function(1)));
};

class __attribute__((shared_trylock_function(1))) StfTestClass { // \
    expected-warning {{'shared_trylock_function' attribute only applies to functions and methods}}
};


/***********************************
 *  Unlock Function (uf)
 ***********************************/

#if !__has_attribute(unlock_function)
#error "Should support unlock_function attribute"
#endif

// takes zero or more arguments, all locks (vars/fields)

void uf_function() __attribute__((unlock_function));

void uf_function_args() __attribute__((unlock_function(mu1, mu2)));

int uf_testfn(int y) __attribute__((unlock_function));

int uf_testfn(int y) {
  int x __attribute__((unlock_function)) = y; // \
    expected-warning {{'unlock_function' attribute only applies to functions and methods}}
  return x;
};

int uf_test_var __attribute__((unlock_function)); // \
  expected-warning {{'unlock_function' attribute only applies to functions and methods}}

class UfFoo {
 private:
  int test_field __attribute__((unlock_function)); // \
    expected-warning {{'unlock_function' attribute only applies to functions and methods}}
  void test_method() __attribute__((unlock_function));
};

class __attribute__((no_thread_safety_analysis)) UfTestClass { // \
    expected-warning {{'no_thread_safety_analysis' attribute only applies to functions and methods}}
};

void uf_fun_params(int lvar __attribute__((unlock_function))); // \
  expected-warning {{'unlock_function' attribute only applies to functions and methods}}


/***********************************
 *  Lock Returned (lr)
 ***********************************/

#if !__has_attribute(lock_returned)
#error "Should support lock_returned attribute"
#endif

// Takes exactly one argument, a var/field

void lr_function() __attribute__((lock_returned)); // \
  expected-error {{attribute takes one argument}}

void lr_function_arg() __attribute__((lock_returned(mu1)));

void lr_function_args() __attribute__((lock_returned(mu1, mu2))); // \
  expected-error {{attribute takes one argument}}

int lr_testfn(int y) __attribute__((lock_returned(mu1)));

int lr_testfn(int y) {
  int x __attribute__((lock_returned(mu1))) = y; // \
    expected-warning {{'lock_returned' attribute only applies to functions and methods}}
  return x;
};

int lr_test_var __attribute__((lock_returned(mu1))); // \
  expected-warning {{'lock_returned' attribute only applies to functions and methods}}

void lr_fun_params(int lvar __attribute__((lock_returned(mu1)))); // \
  expected-warning {{'lock_returned' attribute only applies to functions and methods}}

class LrFoo {
 private:
  int test_field __attribute__((lock_returned(mu1))); // \
    expected-warning {{'lock_returned' attribute only applies to functions and methods}}
  void test_method() __attribute__((lock_returned(mu1)));
};

class __attribute__((lock_returned(mu1))) LrTestClass { // \
    expected-warning {{'lock_returned' attribute only applies to functions and methods}}
};

/***********************************
 *  Locks Excluded (le)
 ***********************************/

#if !__has_attribute(locks_excluded)
#error "Should support locks_excluded attribute"
#endif

// takes one or more arguments, all locks (vars/fields)

void le_function() __attribute__((locks_excluded)); // \
  expected-error {{attribute takes at least 1 argument}}

void le_function_arg() __attribute__((locks_excluded(mu1)));

void le_function_args() __attribute__((locks_excluded(mu1, mu2)));

int le_testfn(int y) __attribute__((locks_excluded(mu1)));

int le_testfn(int y) {
  int x __attribute__((locks_excluded(mu1))) = y; // \
    expected-warning {{'locks_excluded' attribute only applies to functions and methods}}
  return x;
};

int le_test_var __attribute__((locks_excluded(mu1))); // \
  expected-warning {{'locks_excluded' attribute only applies to functions and methods}}

void le_fun_params(int lvar __attribute__((locks_excluded(mu1)))); // \
  expected-warning {{'locks_excluded' attribute only applies to functions and methods}}

class LeFoo {
 private:
  int test_field __attribute__((locks_excluded(mu1))); // \
    expected-warning {{'locks_excluded' attribute only applies to functions and methods}}
  void test_method() __attribute__((locks_excluded(mu1)));
};

class __attribute__((locks_excluded(mu1))) LeTestClass { // \
    expected-warning {{'locks_excluded' attribute only applies to functions and methods}}
};


/***********************************
 *  Exclusive Locks Required (elr)
 ***********************************/

#if !__has_attribute(exclusive_locks_required)
#error "Should support exclusive_locks_required attribute"
#endif

// takes one or more arguments, all locks (vars/fields)

void elr_function() __attribute__((exclusive_locks_required)); // \
  expected-error {{attribute takes at least 1 argument}}

void elr_function_arg() __attribute__((exclusive_locks_required(mu1)));

void elr_function_args() __attribute__((exclusive_locks_required(mu1, mu2)));

int elr_testfn(int y) __attribute__((exclusive_locks_required(mu1)));

int elr_testfn(int y) {
  int x __attribute__((exclusive_locks_required(mu1))) = y; // \
    expected-warning {{'exclusive_locks_required' attribute only applies to functions and methods}}
  return x;
};

int elr_test_var __attribute__((exclusive_locks_required(mu1))); // \
  expected-warning {{'exclusive_locks_required' attribute only applies to functions and methods}}

void elr_fun_params(int lvar __attribute__((exclusive_locks_required(mu1)))); // \
  expected-warning {{'exclusive_locks_required' attribute only applies to functions and methods}}

class ElrFoo {
 private:
  int test_field __attribute__((exclusive_locks_required(mu1))); // \
    expected-warning {{'exclusive_locks_required' attribute only applies to functions and methods}}
  void test_method() __attribute__((exclusive_locks_required(mu1)));
};

class __attribute__((exclusive_locks_required(mu1))) ElrTestClass { // \
    expected-warning {{'exclusive_locks_required' attribute only applies to functions and methods}}
};

/***********************************
 *  Shared Locks Required (slr)
 ***********************************/

#if !__has_attribute(shared_locks_required)
#error "Should support shared_locks_required attribute"
#endif

// takes one or more arguments, all locks (vars/fields)

void slr_function() __attribute__((shared_locks_required)); // \
  expected-error {{attribute takes at least 1 argument}}

void slr_function_arg() __attribute__((shared_locks_required(mu1)));

void slr_function_args() __attribute__((shared_locks_required(mu1, mu2)));

int slr_testfn(int y) __attribute__((shared_locks_required(mu1)));

int slr_testfn(int y) {
  int x __attribute__((shared_locks_required(mu1))) = y; // \
    expected-warning {{'shared_locks_required' attribute only applies to functions and methods}}
  return x;
};

int slr_test_var __attribute__((shared_locks_required(mu1))); // \
  expected-warning {{'shared_locks_required' attribute only applies to functions and methods}}

void slr_fun_params(int lvar __attribute__((shared_locks_required(mu1)))); // \
  expected-warning {{'shared_locks_required' attribute only applies to functions and methods}}

class SlrFoo {
 private:
  int test_field __attribute__((shared_locks_required(mu1))); // \
    expected-warning {{'shared_locks_required' attribute only applies to functions and methods}}
  void test_method() __attribute__((shared_locks_required(mu1)));
};

class __attribute__((shared_locks_required(mu1))) SlrTestClass { // \
    expected-warning {{'shared_locks_required' attribute only applies to functions and methods}}
};
