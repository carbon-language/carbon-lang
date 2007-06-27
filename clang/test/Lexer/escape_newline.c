// RUN: clang -E %s | grep -- ' ->' &&
// RUN: clang -parse-ast-check %s

// This is an ugly way to spell a -> token.
/* expected-warning {{trigraph converted to '\' character}} \
   expected-warning {{backslash and newline separated by space}} \
   expected-error {{expected identifier or '('}} */ -??/      
>
