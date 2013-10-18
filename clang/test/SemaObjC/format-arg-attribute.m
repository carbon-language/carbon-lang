// RUN: %clang_cc1 -Werror -Wformat-nonliteral -verify -fsyntax-only %s

@class NSString;

extern NSString *fa2 (const NSString *) __attribute__((format_arg(1)));
extern NSString *fa3 (NSString *) __attribute__((format_arg(1)));

extern void fc1 (const NSString *) __attribute__((format_arg));  // expected-error {{'format_arg' attribute takes one argument}}
extern void fc2 (const NSString *) __attribute__((format_arg())); // expected-error {{'format_arg' attribute takes one argument}}
extern void fc3 (const NSString *) __attribute__((format_arg(1, 2))); // expected-error {{'format_arg' attribute takes one argument}}

struct s1 { int i; } __attribute__((format_arg(1)));  // expected-error {{'format_arg' attribute only applies to functions}}
union u1 { int i; } __attribute__((format_arg(1)));  // expected-error {{'format_arg' attribute only applies to functions}}
enum e1 { E1V0 } __attribute__((format_arg(1))); // expected-error {{'format_arg' attribute only applies to functions}}

extern NSString *ff3 (const NSString *) __attribute__((format_arg(3-2)));
extern NSString *ff4 (const NSString *) __attribute__((format_arg(foo))); // expected-error {{use of undeclared identifier 'foo'}}

/* format_arg formats must take and return a string.  */
extern NSString *fi0 (int) __attribute__((format_arg(1)));  // expected-error {{format argument not a string type}}
extern NSString *fi1 (NSString *) __attribute__((format_arg(1))); 

extern NSString *fi2 (NSString *) __attribute__((format_arg(1))); 

extern int fi3 (const NSString *) __attribute__((format_arg(1)));  // expected-error {{function does not return NSString}}
extern NSString *fi4 (const NSString *) __attribute__((format_arg(1))); 
extern NSString *fi5 (const NSString *) __attribute__((format_arg(1))); 

// rdar://15242010
@interface NSString
+ (id)stringWithFormat:(NSString *)format, ... __attribute__((format(__NSString__, 1, 2)));
@end

@interface NSBundle
- (NSString *)localizedStringForKey:(NSString *)key value:(NSString *)value table:(NSString *)tableName __attribute__ ((format_arg(1)));
+ (NSBundle *)mainBundle;
@end


NSString* localizedFormat(NSString* string) __attribute__ ((format_arg(1)));

int main()
{
  [NSString stringWithFormat:[[NSBundle mainBundle] localizedStringForKey:@"foo %d" value:@"bar %d" table:0], 42];

  [NSString stringWithFormat:localizedFormat(@"foo %d"), 42];
}
