; Test forward references and redefinitions of globals

%Y = global void()* %X

%A = global int* %B
%B = global int 7
%B = global int 7


declare void %X()

declare void %X()

void %X() {
  ret void
}

declare void %X()
