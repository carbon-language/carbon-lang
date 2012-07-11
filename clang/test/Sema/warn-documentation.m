// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -Wdocumentation-pedantic -verify %s

@class NSString;

// expected-warning@+2 {{empty paragraph passed to '\brief' command}}
/**
 * \brief\brief Aaa
 */
@interface A
// expected-warning@+2 {{empty paragraph passed to '\brief' command}}
/**
 * \brief\brief Aaa
 * \param aaa Aaa
 * \param bbb Bbb
 */
+ (NSString *)test1:(NSString *)aaa suffix:(NSString *)bbb;

// expected-warning@+2 {{parameter 'aab' not found in the function declaration}} expected-note@+2 {{did you mean 'aaa'?}}
/**
 * \param aab Aaa
 */
+ (NSString *)test2:(NSString *)aaa;
@end

