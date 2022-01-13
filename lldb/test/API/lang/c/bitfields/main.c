#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main (int argc, char const *argv[])
{
    struct Bits
    {
        uint32_t    : 1, // Unnamed bitfield
                    b1 : 1,
                    b2 : 2,
                    : 2, // Unnamed bitfield
                    b3 : 3,
                    : 2, // Unnamed bitfield (this will get removed)
                    b4 __attribute__ ((aligned(16))),
                    b5 : 5,
                    b6 : 6,
                    b7 : 7,
                    four : 4;
    };

    printf("%lu", sizeof(struct Bits));

    struct Bits bits;
    int i;
    for (i=0; i<(1<<1); i++)
        bits.b1 = i;        //// break $source:$line
    for (i=0; i<(1<<2); i++)
        bits.b2 = i;        //// break $source:$line
    for (i=0; i<(1<<3); i++)
        bits.b3 = i;        //// break $source:$line
    for (i=0; i<(1<<4); i++)
        bits.b4 = i;        //// break $source:$line
    for (i=0; i<(1<<5); i++)
        bits.b5 = i;        //// break $source:$line
    for (i=0; i<(1<<6); i++)
        bits.b6 = i;        //// break $source:$line
    for (i=0; i<(1<<7); i++)
        bits.b7 = i;        //// break $source:$line
    for (i=0; i<(1<<4); i++)
        bits.four = i;      //// break $source:$line

    struct MoreBits
    {
        uint32_t    a : 3;
        uint8_t       : 1;
        uint8_t     b : 1;
        uint8_t     c : 1;
        uint8_t     d : 1;
    };

    struct MoreBits more_bits;

    more_bits.a = 3;
    more_bits.b = 0;
    more_bits.c = 1;
    more_bits.d = 0;

    struct EvenMoreBits
    {
        uint8_t b1  : 1, b2  : 1, b3  : 1, b4  : 1, b5  : 1, b6  : 1,
                b7  : 1, b8  : 1, b9  : 1, b10 : 1, b11 : 1, b12 : 1,
                b13 : 1, b14 : 1, b15 : 1, b16 : 1, b17 : 1;
    };

    struct EvenMoreBits even_more_bits;
    memset(&even_more_bits, 0, sizeof(even_more_bits));
    even_more_bits.b1 = 1;
    even_more_bits.b5 = 1;
    even_more_bits.b7 = 1;
    even_more_bits.b13 = 1;

#pragma pack(1)
    struct PackedBits
    {
        char a;
    	uint32_t b : 5,
                 c : 27;
    };
#pragma pack()  
    struct PackedBits packed;
    packed.a = 'a';
    packed.b = 10;
    packed.c = 0x7112233;

    struct LargePackedBits {
        uint64_t a: 36;
        uint64_t b: 36;
    } __attribute__((packed));

    struct LargePackedBits large_packed =
      (struct LargePackedBits){ 0xcbbbbaaaa, 0xdffffeeee };
    struct LargePackedBits *large_packed_ptr = &large_packed;
    
    return 0;               //// Set break point at this line.

}
