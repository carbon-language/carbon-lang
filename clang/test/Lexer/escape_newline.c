// RUN: clang -E %s | grep -- ' ->' &&
// RUN: clang -E %s 2>&1 | grep 'backslash and newline separated by space' &&
// RUN: clang -E %s 2>&1 | grep 'trigraph converted'

// This is an ugly way to spell a -> token.
 -??/      
>
