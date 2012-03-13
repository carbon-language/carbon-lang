// Parse diagnostic arguments in the driver
// PR12181

// RUN: !%clang -target x86_64-apple-darwin10 \
// RUN:   -fsyntax-only -fzyzzybalubah \ 
// RUN:   -Werror=unused-command-line-argument %s 

// RUN: !%clang -target x86_64-apple-darwin10 \
// RUN:   -fsyntax-only -fzyzzybalubah -Werror %s 
