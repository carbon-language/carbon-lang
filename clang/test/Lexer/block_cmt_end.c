/*
  RUN: clang -E -trigraphs %s | grep bar &&
  RUN: clang -E -trigraphs %s | grep foo &&
  RUN: clang -E -trigraphs %s | not grep abc &&
  RUN: clang -E -trigraphs %s | not grep xyz &&
  RUN: clang -fsyntax-only -trigraphs -verify %s  
*/

// This is a simple comment, /*/ does not end a comment, the trailing */ does.
int i = /*/ */ 1;

/* abc

next comment ends with normal escaped newline:
*/

/* expected-warning {{escaped newline}} expected-warning {{backslash and newline}}  *\  
/

bar

/* xyz

next comment ends with a trigraph escaped newline: */

/* expected-warning {{escaped newline between}}   expected-warning {{backslash and newline separated by space}}    expected-warning {{trigraph ends block comment}}   *??/    
/

foo /* expected-error {{expected '=', ',', ';', 'asm', or '__attribute__' after declarator}} */

