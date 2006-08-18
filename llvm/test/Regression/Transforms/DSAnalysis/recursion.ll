; RUN: opt -analyze %s -tddatastructure

implementation   ; Functions:

declare void %__main()

void %A(int* %L) {
bb0:            ; No predecessors!
        call void %B( int* %L )
        call void %A( int* %L )
        ret void
}

void %B(int* %L) {
bb0:            ; No predecessors!
        call void %A( int* %L )
        ret void
}

void %main() {
bb0:            ; No predecessors!
        call void %__main( )
        call void %A( int* null )
        ret void
}

