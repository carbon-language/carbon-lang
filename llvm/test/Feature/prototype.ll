implementation

declare int "bar"(int %in) 

int "foo"(int %blah)
begin
  %xx = call int(int) %bar(int %blah)
  ret int %xx
end

