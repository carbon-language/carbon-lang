// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

fn identFn( x: i32 ) -> i32{
    return x;
}

fn Main() -> i32 {
  var x : i32 = 7 % 2;


  if(  not (x == 1) ){
      return 1;
  }
  x = (3*3) % 3;
  if( not (x == 0)){
      return 2;
  }

  x = (4+4+4+4) % (2+1);
  if(  not (x == 1) ){
      return 3;
  }

  x = 15 % (identFn(2)+1);
  if( not (x == 0)){
      return 4;
  }

  x = -15 % 7;
  if( not (x == -1)){
      return 5;
  }

  x = 15 % -7;
  if( not ( x == 1)){
      return 6;
  }

  x = -15 % -identFn(7);
  if( not (x == -1)){
      return 7;
  }

  return 0;
}
