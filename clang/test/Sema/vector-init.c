// RUN: clang %s -verify -fsyntax-only

typedef __attribute__(( ext_vector_type(4) ))  float float4;

float4 foo = (float4){ 1.0, 2.0, 3.0, 4.0 };
