// RUN: clang %s -fsyntax-only

// Top level extension marker.

__extension__ typedef struct
{
    long long int quot; 
    long long int rem; 
}lldiv_t; 
