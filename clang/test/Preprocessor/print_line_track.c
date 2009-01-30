/* RUN: clang -E %s | grep 'a 3' &&
 * RUN: clang -E %s | grep 'b 16' &&
 * RUN: clang -E -P %s | grep 'a 3' &&
 * RUN: clang -E -P %s | grep 'b 16' &&
 * RUN: clang -E %s | not grep '# 0 '
 * PR1848
 * PR3437
*/

#define t(x) x

t(a
3)

t(b
__LINE__)

