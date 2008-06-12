/* RUN: clang -E %s | grep 'a 3' &&
 * RUN: clang -E %s | grep 'b 14' &&
 * RUN: clang -E -P %s | grep 'a 3' &&
 * RUN: clang -E -P %s | grep 'b 14'
 * PR1848
*/

#define t(x) x

t(a
3)

t(b
__LINE__)

