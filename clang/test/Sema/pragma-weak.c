// RUN: clang-cc -emit-llvm -o %t %s &&

#pragma weak weakvar
int weakvar;
// RUN: grep '@weakvar = weak global' %t | count 1 &&

#pragma weak weakdef
void weakdef(void) {}
// RUN: grep 'define weak void @weakdef() nounwind {' %t | count 1 &&

#pragma weak param // expected-warning {{weak identifier 'param' never declared}}
#pragma weak correct_linkage
void f(int param) {
  int correct_linkage;
}
int correct_linkage;
// RUN: grep '@correct_linkage = weak global' %t | count 1 &&

#pragma weak weakvar_alias = __weakvar_alias
int __weakvar_alias;
// RUN: grep '@__weakvar_alias = common global' %t | count 1 &&
// RUN: grep '@weakvar_alias = alias weak i32\* @__weakvar_alias' %t | count 1 &&
//@weakvar_alias = alias weak i32* @__weakvar_alias

#pragma weak foo = __foo
void __foo(void) {}
// RUN: grep '@foo = alias weak void ()\* @__foo\>' %t | count 1 &&
// RUN: grep 'define void @__foo() nounwind {' %t | count 1 &&


void __foo2(void) {}
#pragma weak foo2 = __foo2
// RUN: grep '@foo2 = alias weak void ()\* @__foo2\>' %t | count 1 &&
// RUN: grep 'define void @__foo2() nounwind {' %t | count 1 &&


///// test errors

#pragma weak unused // expected-warning {{weak identifier 'unused' never declared}}
#pragma weak unused_alias = __unused_alias  // expected-warning {{weak identifier '__unused_alias' never declared}}

#pragma weak td // expected-warning {{weak identifier 'td' never declared}}
typedef int td;

#pragma weak td2 = __td2 // expected-warning {{weak identifier '__td2' never declared}}
typedef int __td2;


///// test weird cases

// test repeats

#pragma weak stutter = __stutter
#pragma weak stutter = __stutter
void __stutter(void) {}
// RUN: grep '@stutter = alias weak void ()\* @__stutter\>' %t | count 1 &&
// RUN: grep 'define void @__stutter() nounwind {' %t | count 1 &&

void __stutter2(void) {}
#pragma weak stutter2 = __stutter2
#pragma weak stutter2 = __stutter2
// RUN: grep '@stutter2 = alias weak void ()\* @__stutter2\>' %t | count 1 &&
// RUN: grep 'define void @__stutter2() nounwind {' %t | count 1 &&


// test decl/pragma weak order

void __declfirst(void);
#pragma weak declfirst = __declfirst
void __declfirst(void) {}
// RUN: grep '@declfirst = alias weak void ()\* @__declfirst\>' %t | count 1 &&
// RUN: grep 'define void @__declfirst() nounwind {' %t | count 1 &&

void __declfirstattr(void) __attribute((noinline));
#pragma weak declfirstattr = __declfirstattr
void __declfirstattr(void) {}
// RUN: grep '@declfirstattr = alias weak void ()\* @__declfirstattr\>' %t | count 1 &&
// RUN: grep 'define void @__declfirstattr() nounwind noinline {' %t | count 1 &&

//// test that other attributes are preserved

//// ensure that pragma weak/__attribute((weak)) play nice

void mix(void);
#pragma weak mix
__attribute((weak)) void mix(void) { }
// RUN: grep 'define weak void @mix() nounwind {' %t | count 1 &&

// ensure following __attributes are preserved and that only a single
// alias is generated
#pragma weak mix2 = __mix2
void __mix2(void) __attribute((noinline));
void __mix2(void) __attribute((noinline));
void __mix2(void) {}
// RUN: grep '@mix2 = alias weak void ()\* @__mix2\>' %t | count 1 &&
// RUN: grep 'define void @__mix2() nounwind noinline {' %t | count 1 &&

////////////// test #pragma weak/__attribute combinations

// if the SAME ALIAS is already declared then it overrides #pragma weak
// resulting in a non-weak alias in this case
void both(void) __attribute((alias("__both")));
#pragma weak both = __both
void __both(void) {}
// RUN: grep '@both = alias void ()\* @__both\>' %t | count 1 &&
// RUN: grep 'define void @__both() nounwind {' %t | count 1 &&

// if the TARGET is previously declared then whichever aliasing method
// comes first applies and subsequent aliases are discarded.
// TODO: warn about this

void __both2(void);
void both2(void) __attribute((alias("__both2"))); // first, wins
#pragma weak both2 = __both2
void __both2(void) {}
// RUN: grep '@both2 = alias void ()\* @__both2\>' %t | count 1 &&
// RUN: grep 'define void @__both2() nounwind {' %t | count 1 &&

void __both3(void);
#pragma weak both3 = __both3 // first, wins
void both3(void) __attribute((alias("__both3")));
void __both3(void) {}
// RUN: grep '@both3 = alias weak void ()\* @__both3\>' %t | count 1 &&
// RUN: grep 'define void @__both3() nounwind {' %t | count 1 &&

///////////// ensure that #pragma weak does not alter existing __attributes()

void __a1(void) __attribute((noinline));
#pragma weak a1 = __a1
void __a1(void) {}
// RUN: grep '@a1 = alias weak void ()\* @__a1\>' %t | count 1 &&
// RUN: grep 'define void @__a1() nounwind noinline {' %t | count 1 &&

// attributes introduced BEFORE a combination of #pragma weak and alias()
// hold...
void __a3(void) __attribute((noinline));
#pragma weak a3 = __a3
void a3(void) __attribute((alias("__a3")));
void __a3(void) {}
// RUN: grep '@a3 = alias weak void ()\* @__a3\>' %t | count 1 &&
// RUN: grep 'define void @__a3() nounwind noinline {' %t | count 1 &&

#pragma weak xxx = __xxx
__attribute((pure,noinline,const,fastcall)) void __xxx(void) { }
// RUN: grep '@xxx = alias weak void ()\* @__xxx\>' %t | count 1 &&
// RUN: grep 'define .*fastcall.* void @__xxx() nounwind readnone noinline {' %t | count 1 &&

/// TODO: stuff that still doesn't work

// due to the fact that disparate TopLevelDecls cannot affect each other
// (due to clang's Parser and ASTConsumer behavior, and quite reasonable)
// #pragma weak must appear before or within the same TopLevelDecl as it
// references.
void yyy(void){}
void zzz(void){}
#pragma weak yyy
// NOTE: weak doesn't apply, not before or in same TopLevelDec(!)
// RUN: grep 'define void @yyy() nounwind {' %t | count 1
