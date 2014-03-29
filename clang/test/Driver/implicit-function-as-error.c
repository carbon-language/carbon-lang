// RUN: %clang -target x86_64-apple-darwin -mios-simulator-version-min=7 -fsyntax-only %s -Xclang -verify
// RUN: %clang -target x86_64-apple-darwin -arch arm64 -target x86_64-apple-darwin -mios-version-min=7 -fsyntax-only %s -Xclang -verify

// For 64-bit iOS, automatically promote -Wimplicit-function-declaration
// to an error.

void radar_10894044() {
  radar_10894044_not_declared(); // expected-error {{implicit declaration of function 'radar_10894044_not_declared' is invalid in C99}}
}
