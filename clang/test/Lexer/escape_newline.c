// RUN: clang-cc -E -trigraphs %s | grep -- ' ->'
// RUN: clang-cc -E -trigraphs %s 2>&1 | grep 'backslash and newline separated by space'
// RUN: clang-cc -E -trigraphs %s 2>&1 | grep 'trigraph converted'

// This is an ugly way to spell a -> token.
 -??/      
>
