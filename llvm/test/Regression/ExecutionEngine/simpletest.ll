implementation

int %bar() { ret int 0 }

int %main() {
        %r = call int %bar()
        ret int %r
}

