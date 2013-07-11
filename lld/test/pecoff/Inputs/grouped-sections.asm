.386
.model flat, c

_data$2 SEGMENT BYTE alias(".data$2")
	db "orld", 0
_data$2 ends

_data$1 SEGMENT BYTE alias(".data$1")
	db "o, w"
_data$1 ends

.data
	db "Hell"

.code
main:
	nop
end main
