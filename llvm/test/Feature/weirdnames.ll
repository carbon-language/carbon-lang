; Test using double quotes to form names that are not legal in the % form

"&^ " = type { int }
"%.*+ foo" = global "&^ " { int 5 }
"0" = global float 0.0                 ; This CANNOT be %0
