%g = global int 0

int %main() {  
  %h = load int* %g
  ret int %h
}
