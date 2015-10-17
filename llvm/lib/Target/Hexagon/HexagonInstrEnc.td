class Enc_COPROC_VX_3op_v<bits<15> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<5> src1;
  bits<5> src2;

  let Inst{31-16} = { opc{14-4}, src2};
  let Inst{13-0} = { opc{3}, src1, opc{2-0}, dst};
}

class V6_vtmpyb_enc : Enc_COPROC_VX_3op_v<0b000110010000000>;
class V6_vtmpybus_enc : Enc_COPROC_VX_3op_v<0b000110010000001>;
class V6_vdmpyhb_enc : Enc_COPROC_VX_3op_v<0b000110010000010>;
class V6_vrmpyub_enc : Enc_COPROC_VX_3op_v<0b000110010000011>;
class V6_vrmpybus_enc : Enc_COPROC_VX_3op_v<0b000110010000100>;
class V6_vdsaduh_enc : Enc_COPROC_VX_3op_v<0b000110010000101>;
class V6_vdmpybus_enc : Enc_COPROC_VX_3op_v<0b000110010000110>;
class V6_vdmpybus_dv_enc : Enc_COPROC_VX_3op_v<0b000110010000111>;
class V6_vtmpyb_acc_enc : Enc_COPROC_VX_3op_v<0b000110010001000>;
class V6_vtmpybus_acc_enc : Enc_COPROC_VX_3op_v<0b000110010001001>;
class V6_vtmpyhb_acc_enc : Enc_COPROC_VX_3op_v<0b000110010001010>;
class V6_vdmpyhb_acc_enc : Enc_COPROC_VX_3op_v<0b000110010001011>;
class V6_vrmpyub_acc_enc : Enc_COPROC_VX_3op_v<0b000110010001100>;
class V6_vrmpybus_acc_enc : Enc_COPROC_VX_3op_v<0b000110010001101>;
class V6_vdmpybus_acc_enc : Enc_COPROC_VX_3op_v<0b000110010001110>;
class V6_vdmpybus_dv_acc_enc : Enc_COPROC_VX_3op_v<0b000110010001111>;
class V6_vdmpyhsusat_enc : Enc_COPROC_VX_3op_v<0b000110010010000>;
class V6_vdmpyhsuisat_enc : Enc_COPROC_VX_3op_v<0b000110010010001>;
class V6_vdmpyhsat_enc : Enc_COPROC_VX_3op_v<0b000110010010010>;
class V6_vdmpyhisat_enc : Enc_COPROC_VX_3op_v<0b000110010010011>;
class V6_vdmpyhb_dv_enc : Enc_COPROC_VX_3op_v<0b000110010010100>;
class V6_vmpybus_enc : Enc_COPROC_VX_3op_v<0b000110010010101>;
class V6_vmpabus_enc : Enc_COPROC_VX_3op_v<0b000110010010110>;
class V6_vmpahb_enc : Enc_COPROC_VX_3op_v<0b000110010010111>;
class V6_vdmpyhsusat_acc_enc : Enc_COPROC_VX_3op_v<0b000110010011000>;
class V6_vdmpyhsuisat_acc_enc : Enc_COPROC_VX_3op_v<0b000110010011001>;
class V6_vdmpyhisat_acc_enc : Enc_COPROC_VX_3op_v<0b000110010011010>;
class V6_vdmpyhsat_acc_enc : Enc_COPROC_VX_3op_v<0b000110010011011>;
class V6_vdmpyhb_dv_acc_enc : Enc_COPROC_VX_3op_v<0b000110010011100>;
class V6_vmpybus_acc_enc : Enc_COPROC_VX_3op_v<0b000110010011101>;
class V6_vmpabus_acc_enc : Enc_COPROC_VX_3op_v<0b000110010011110>;
class V6_vmpahb_acc_enc : Enc_COPROC_VX_3op_v<0b000110010011111>;
class V6_vmpyh_enc : Enc_COPROC_VX_3op_v<0b000110010100000>;
class V6_vmpyhss_enc : Enc_COPROC_VX_3op_v<0b000110010100001>;
class V6_vmpyhsrs_enc : Enc_COPROC_VX_3op_v<0b000110010100010>;
class V6_vmpyuh_enc : Enc_COPROC_VX_3op_v<0b000110010100011>;
class V6_vmpyhsat_acc_enc : Enc_COPROC_VX_3op_v<0b000110010101000>;
class V6_vmpyuh_acc_enc : Enc_COPROC_VX_3op_v<0b000110010101001>;
class V6_vmpyiwb_acc_enc : Enc_COPROC_VX_3op_v<0b000110010101010>;
class V6_vmpyiwh_acc_enc : Enc_COPROC_VX_3op_v<0b000110010101011>;
class V6_vmpyihb_enc : Enc_COPROC_VX_3op_v<0b000110010110000>;
class V6_vror_enc : Enc_COPROC_VX_3op_v<0b000110010110001>;
class V6_vasrw_enc : Enc_COPROC_VX_3op_v<0b000110010110101>;
class V6_vasrh_enc : Enc_COPROC_VX_3op_v<0b000110010110110>;
class V6_vaslw_enc : Enc_COPROC_VX_3op_v<0b000110010110111>;
class V6_vdsaduh_acc_enc : Enc_COPROC_VX_3op_v<0b000110010111000>;
class V6_vmpyihb_acc_enc : Enc_COPROC_VX_3op_v<0b000110010111001>;
class V6_vaslw_acc_enc : Enc_COPROC_VX_3op_v<0b000110010111010>;
class V6_vasrw_acc_enc : Enc_COPROC_VX_3op_v<0b000110010111101>;
class V6_vaslh_enc : Enc_COPROC_VX_3op_v<0b000110011000000>;
class V6_vlsrw_enc : Enc_COPROC_VX_3op_v<0b000110011000001>;
class V6_vlsrh_enc : Enc_COPROC_VX_3op_v<0b000110011000010>;
class V6_vmpyiwh_enc : Enc_COPROC_VX_3op_v<0b000110011000111>;
class V6_vmpyub_acc_enc : Enc_COPROC_VX_3op_v<0b000110011001000>;
class V6_vmpyiwb_enc : Enc_COPROC_VX_3op_v<0b000110011010000>;
class V6_vtmpyhb_enc : Enc_COPROC_VX_3op_v<0b000110011010100>;
class V6_vmpyub_enc : Enc_COPROC_VX_3op_v<0b000110011100000>;
class V6_vrmpyubv_enc : Enc_COPROC_VX_3op_v<0b000111000000000>;
class V6_vrmpybv_enc : Enc_COPROC_VX_3op_v<0b000111000000001>;
class V6_vrmpybusv_enc : Enc_COPROC_VX_3op_v<0b000111000000010>;
class V6_vdmpyhvsat_enc : Enc_COPROC_VX_3op_v<0b000111000000011>;
class V6_vmpybv_enc : Enc_COPROC_VX_3op_v<0b000111000000100>;
class V6_vmpyubv_enc : Enc_COPROC_VX_3op_v<0b000111000000101>;
class V6_vmpybusv_enc : Enc_COPROC_VX_3op_v<0b000111000000110>;
class V6_vmpyhv_enc : Enc_COPROC_VX_3op_v<0b000111000000111>;
class V6_vrmpyubv_acc_enc : Enc_COPROC_VX_3op_v<0b000111000001000>;
class V6_vrmpybv_acc_enc : Enc_COPROC_VX_3op_v<0b000111000001001>;
class V6_vrmpybusv_acc_enc : Enc_COPROC_VX_3op_v<0b000111000001010>;
class V6_vdmpyhvsat_acc_enc : Enc_COPROC_VX_3op_v<0b000111000001011>;
class V6_vmpybv_acc_enc : Enc_COPROC_VX_3op_v<0b000111000001100>;
class V6_vmpyubv_acc_enc : Enc_COPROC_VX_3op_v<0b000111000001101>;
class V6_vmpybusv_acc_enc : Enc_COPROC_VX_3op_v<0b000111000001110>;
class V6_vmpyhv_acc_enc : Enc_COPROC_VX_3op_v<0b000111000001111>;
class V6_vmpyuhv_enc : Enc_COPROC_VX_3op_v<0b000111000010000>;
class V6_vmpyhvsrs_enc : Enc_COPROC_VX_3op_v<0b000111000010001>;
class V6_vmpyhus_enc : Enc_COPROC_VX_3op_v<0b000111000010010>;
class V6_vmpabusv_enc : Enc_COPROC_VX_3op_v<0b000111000010011>;
class V6_vmpyih_enc : Enc_COPROC_VX_3op_v<0b000111000010100>;
class V6_vand_enc : Enc_COPROC_VX_3op_v<0b000111000010101>;
class V6_vor_enc : Enc_COPROC_VX_3op_v<0b000111000010110>;
class V6_vxor_enc : Enc_COPROC_VX_3op_v<0b000111000010111>;
class V6_vmpyuhv_acc_enc : Enc_COPROC_VX_3op_v<0b000111000011000>;
class V6_vmpyhus_acc_enc : Enc_COPROC_VX_3op_v<0b000111000011001>;
class V6_vmpyih_acc_enc : Enc_COPROC_VX_3op_v<0b000111000011100>;
class V6_vmpyiewuh_acc_enc : Enc_COPROC_VX_3op_v<0b000111000011101>;
class V6_vmpyowh_sacc_enc : Enc_COPROC_VX_3op_v<0b000111000011110>;
class V6_vmpyowh_rnd_sacc_enc : Enc_COPROC_VX_3op_v<0b000111000011111>;
class V6_vaddw_enc : Enc_COPROC_VX_3op_v<0b000111000100000>;
class V6_vaddubsat_enc : Enc_COPROC_VX_3op_v<0b000111000100001>;
class V6_vadduhsat_enc : Enc_COPROC_VX_3op_v<0b000111000100010>;
class V6_vaddhsat_enc : Enc_COPROC_VX_3op_v<0b000111000100011>;
class V6_vaddwsat_enc : Enc_COPROC_VX_3op_v<0b000111000100100>;
class V6_vsubb_enc : Enc_COPROC_VX_3op_v<0b000111000100101>;
class V6_vsubh_enc : Enc_COPROC_VX_3op_v<0b000111000100110>;
class V6_vsubw_enc : Enc_COPROC_VX_3op_v<0b000111000100111>;
class V6_vmpyiewh_acc_enc : Enc_COPROC_VX_3op_v<0b000111000101000>;
class V6_vsububsat_enc : Enc_COPROC_VX_3op_v<0b000111000110000>;
class V6_vsubuhsat_enc : Enc_COPROC_VX_3op_v<0b000111000110001>;
class V6_vsubhsat_enc : Enc_COPROC_VX_3op_v<0b000111000110010>;
class V6_vsubwsat_enc : Enc_COPROC_VX_3op_v<0b000111000110011>;
class V6_vaddb_dv_enc : Enc_COPROC_VX_3op_v<0b000111000110100>;
class V6_vaddh_dv_enc : Enc_COPROC_VX_3op_v<0b000111000110101>;
class V6_vaddw_dv_enc : Enc_COPROC_VX_3op_v<0b000111000110110>;
class V6_vaddubsat_dv_enc : Enc_COPROC_VX_3op_v<0b000111000110111>;
class V6_vadduhsat_dv_enc : Enc_COPROC_VX_3op_v<0b000111001000000>;
class V6_vaddhsat_dv_enc : Enc_COPROC_VX_3op_v<0b000111001000001>;
class V6_vaddwsat_dv_enc : Enc_COPROC_VX_3op_v<0b000111001000010>;
class V6_vsubb_dv_enc : Enc_COPROC_VX_3op_v<0b000111001000011>;
class V6_vsubh_dv_enc : Enc_COPROC_VX_3op_v<0b000111001000100>;
class V6_vsubw_dv_enc : Enc_COPROC_VX_3op_v<0b000111001000101>;
class V6_vsububsat_dv_enc : Enc_COPROC_VX_3op_v<0b000111001000110>;
class V6_vsubuhsat_dv_enc : Enc_COPROC_VX_3op_v<0b000111001000111>;
class V6_vsubhsat_dv_enc : Enc_COPROC_VX_3op_v<0b000111001010000>;
class V6_vsubwsat_dv_enc : Enc_COPROC_VX_3op_v<0b000111001010001>;
class V6_vaddubh_enc : Enc_COPROC_VX_3op_v<0b000111001010010>;
class V6_vadduhw_enc : Enc_COPROC_VX_3op_v<0b000111001010011>;
class V6_vaddhw_enc : Enc_COPROC_VX_3op_v<0b000111001010100>;
class V6_vsububh_enc : Enc_COPROC_VX_3op_v<0b000111001010101>;
class V6_vsubuhw_enc : Enc_COPROC_VX_3op_v<0b000111001010110>;
class V6_vsubhw_enc : Enc_COPROC_VX_3op_v<0b000111001010111>;
class V6_vabsdiffub_enc : Enc_COPROC_VX_3op_v<0b000111001100000>;
class V6_vabsdiffh_enc : Enc_COPROC_VX_3op_v<0b000111001100001>;
class V6_vabsdiffuh_enc : Enc_COPROC_VX_3op_v<0b000111001100010>;
class V6_vabsdiffw_enc : Enc_COPROC_VX_3op_v<0b000111001100011>;
class V6_vavgub_enc : Enc_COPROC_VX_3op_v<0b000111001100100>;
class V6_vavguh_enc : Enc_COPROC_VX_3op_v<0b000111001100101>;
class V6_vavgh_enc : Enc_COPROC_VX_3op_v<0b000111001100110>;
class V6_vavgw_enc : Enc_COPROC_VX_3op_v<0b000111001100111>;
class V6_vnavgub_enc : Enc_COPROC_VX_3op_v<0b000111001110000>;
class V6_vnavgh_enc : Enc_COPROC_VX_3op_v<0b000111001110001>;
class V6_vnavgw_enc : Enc_COPROC_VX_3op_v<0b000111001110010>;
class V6_vavgubrnd_enc : Enc_COPROC_VX_3op_v<0b000111001110011>;
class V6_vavguhrnd_enc : Enc_COPROC_VX_3op_v<0b000111001110100>;
class V6_vavghrnd_enc : Enc_COPROC_VX_3op_v<0b000111001110101>;
class V6_vavgwrnd_enc : Enc_COPROC_VX_3op_v<0b000111001110110>;
class V6_vmpabuuv_enc : Enc_COPROC_VX_3op_v<0b000111001110111>;
class V6_vminub_enc : Enc_COPROC_VX_3op_v<0b000111110000001>;
class V6_vminuh_enc : Enc_COPROC_VX_3op_v<0b000111110000010>;
class V6_vminh_enc : Enc_COPROC_VX_3op_v<0b000111110000011>;
class V6_vminw_enc : Enc_COPROC_VX_3op_v<0b000111110000100>;
class V6_vmaxub_enc : Enc_COPROC_VX_3op_v<0b000111110000101>;
class V6_vmaxuh_enc : Enc_COPROC_VX_3op_v<0b000111110000110>;
class V6_vmaxh_enc : Enc_COPROC_VX_3op_v<0b000111110000111>;
class V6_vmaxw_enc : Enc_COPROC_VX_3op_v<0b000111110010000>;
class V6_vdelta_enc : Enc_COPROC_VX_3op_v<0b000111110010001>;
class V6_vrdelta_enc : Enc_COPROC_VX_3op_v<0b000111110010011>;
class V6_vdealb4w_enc : Enc_COPROC_VX_3op_v<0b000111110010111>;
class V6_vmpyowh_rnd_enc : Enc_COPROC_VX_3op_v<0b000111110100000>;
class V6_vshuffeb_enc : Enc_COPROC_VX_3op_v<0b000111110100001>;
class V6_vshuffob_enc : Enc_COPROC_VX_3op_v<0b000111110100010>;
class V6_vshufeh_enc : Enc_COPROC_VX_3op_v<0b000111110100011>;
class V6_vshufoh_enc : Enc_COPROC_VX_3op_v<0b000111110100100>;
class V6_vshufoeh_enc : Enc_COPROC_VX_3op_v<0b000111110100101>;
class V6_vshufoeb_enc : Enc_COPROC_VX_3op_v<0b000111110100110>;
class V6_vcombine_enc : Enc_COPROC_VX_3op_v<0b000111110100111>;
class V6_vmpyieoh_enc : Enc_COPROC_VX_3op_v<0b000111110110000>;
class V6_vsathub_enc : Enc_COPROC_VX_3op_v<0b000111110110010>;
class V6_vsatwh_enc : Enc_COPROC_VX_3op_v<0b000111110110011>;
class V6_vroundwh_enc : Enc_COPROC_VX_3op_v<0b000111110110100>;
class V6_vroundwuh_enc : Enc_COPROC_VX_3op_v<0b000111110110101>;
class V6_vroundhb_enc : Enc_COPROC_VX_3op_v<0b000111110110110>;
class V6_vroundhub_enc : Enc_COPROC_VX_3op_v<0b000111110110111>;
class V6_vasrwv_enc : Enc_COPROC_VX_3op_v<0b000111111010000>;
class V6_vlsrwv_enc : Enc_COPROC_VX_3op_v<0b000111111010001>;
class V6_vlsrhv_enc : Enc_COPROC_VX_3op_v<0b000111111010010>;
class V6_vasrhv_enc : Enc_COPROC_VX_3op_v<0b000111111010011>;
class V6_vaslwv_enc : Enc_COPROC_VX_3op_v<0b000111111010100>;
class V6_vaslhv_enc : Enc_COPROC_VX_3op_v<0b000111111010101>;
class V6_vaddb_enc : Enc_COPROC_VX_3op_v<0b000111111010110>;
class V6_vaddh_enc : Enc_COPROC_VX_3op_v<0b000111111010111>;
class V6_vmpyiewuh_enc : Enc_COPROC_VX_3op_v<0b000111111100000>;
class V6_vmpyiowh_enc : Enc_COPROC_VX_3op_v<0b000111111100001>;
class V6_vpackeb_enc : Enc_COPROC_VX_3op_v<0b000111111100010>;
class V6_vpackeh_enc : Enc_COPROC_VX_3op_v<0b000111111100011>;
class V6_vpackhub_sat_enc : Enc_COPROC_VX_3op_v<0b000111111100101>;
class V6_vpackhb_sat_enc : Enc_COPROC_VX_3op_v<0b000111111100110>;
class V6_vpackwuh_sat_enc : Enc_COPROC_VX_3op_v<0b000111111100111>;
class V6_vpackwh_sat_enc : Enc_COPROC_VX_3op_v<0b000111111110000>;
class V6_vpackob_enc : Enc_COPROC_VX_3op_v<0b000111111110001>;
class V6_vpackoh_enc : Enc_COPROC_VX_3op_v<0b000111111110010>;
class V6_vmpyewuh_enc : Enc_COPROC_VX_3op_v<0b000111111110101>;
class V6_vmpyowh_enc : Enc_COPROC_VX_3op_v<0b000111111110111>;
class V6_extractw_enc : Enc_COPROC_VX_3op_v<0b100100100000001>;
class M6_vabsdiffub_enc : Enc_COPROC_VX_3op_v<0b111010001010000>;
class M6_vabsdiffb_enc : Enc_COPROC_VX_3op_v<0b111010001110000>;

class Enc_COPROC_VX_cmp<bits<13> opc> : OpcodeHexagon {
  bits<2> dst;
  bits<5> src1;
  bits<5> src2;

  let Inst{31-16} = { 0b00011, opc{12-7}, src2{4-0} };
  let Inst{13-0} = { opc{6}, src1{4-0}, opc{5-0}, dst{1-0} };
}

class V6_vandvrt_acc_enc : Enc_COPROC_VX_cmp<0b0010111100000>;
class V6_vandvrt_enc : Enc_COPROC_VX_cmp<0b0011010010010>;
class V6_veqb_and_enc : Enc_COPROC_VX_cmp<0b1001001000000>;
class V6_veqh_and_enc : Enc_COPROC_VX_cmp<0b1001001000001>;
class V6_veqw_and_enc : Enc_COPROC_VX_cmp<0b1001001000010>;
class V6_vgtb_and_enc : Enc_COPROC_VX_cmp<0b1001001000100>;
class V6_vgth_and_enc : Enc_COPROC_VX_cmp<0b1001001000101>;
class V6_vgtw_and_enc : Enc_COPROC_VX_cmp<0b1001001000110>;
class V6_vgtub_and_enc : Enc_COPROC_VX_cmp<0b1001001001000>;
class V6_vgtuh_and_enc : Enc_COPROC_VX_cmp<0b1001001001001>;
class V6_vgtuw_and_enc : Enc_COPROC_VX_cmp<0b1001001001010>;
class V6_veqb_or_enc : Enc_COPROC_VX_cmp<0b1001001010000>;
class V6_veqh_or_enc : Enc_COPROC_VX_cmp<0b1001001010001>;
class V6_veqw_or_enc : Enc_COPROC_VX_cmp<0b1001001010010>;
class V6_vgtb_or_enc : Enc_COPROC_VX_cmp<0b1001001010100>;
class V6_vgth_or_enc : Enc_COPROC_VX_cmp<0b1001001010101>;
class V6_vgtw_or_enc : Enc_COPROC_VX_cmp<0b1001001010110>;
class V6_vgtub_or_enc : Enc_COPROC_VX_cmp<0b1001001011000>;
class V6_vgtuh_or_enc : Enc_COPROC_VX_cmp<0b1001001011001>;
class V6_vgtuw_or_enc : Enc_COPROC_VX_cmp<0b1001001011010>;
class V6_veqb_xor_enc : Enc_COPROC_VX_cmp<0b1001001100000>;
class V6_veqh_xor_enc : Enc_COPROC_VX_cmp<0b1001001100001>;
class V6_veqw_xor_enc : Enc_COPROC_VX_cmp<0b1001001100010>;
class V6_vgtb_xor_enc : Enc_COPROC_VX_cmp<0b1001001100100>;
class V6_vgth_xor_enc : Enc_COPROC_VX_cmp<0b1001001100101>;
class V6_vgtw_xor_enc : Enc_COPROC_VX_cmp<0b1001001100110>;
class V6_vgtub_xor_enc : Enc_COPROC_VX_cmp<0b1001001101000>;
class V6_vgtuh_xor_enc : Enc_COPROC_VX_cmp<0b1001001101001>;
class V6_vgtuw_xor_enc : Enc_COPROC_VX_cmp<0b1001001101010>;
class V6_veqb_enc : Enc_COPROC_VX_cmp<0b1111000000000>;
class V6_veqh_enc : Enc_COPROC_VX_cmp<0b1111000000001>;
class V6_veqw_enc : Enc_COPROC_VX_cmp<0b1111000000010>;
class V6_vgtb_enc : Enc_COPROC_VX_cmp<0b1111000000100>;
class V6_vgth_enc : Enc_COPROC_VX_cmp<0b1111000000101>;
class V6_vgtw_enc : Enc_COPROC_VX_cmp<0b1111000000110>;
class V6_vgtub_enc : Enc_COPROC_VX_cmp<0b1111000001000>;
class V6_vgtuh_enc : Enc_COPROC_VX_cmp<0b1111000001001>;
class V6_vgtuw_enc : Enc_COPROC_VX_cmp<0b1111000001010>;

class Enc_COPROC_VX_p2op<bits<5> opc> : OpcodeHexagon {
  bits<2> src1;
  bits<5> dst;
  bits<5> src2;

  let Inst{31-16} = { 0b00011110, src1{1-0}, 0b0000, opc{4-3} };
  let Inst{13-0} = { 1, src2{4-0}, opc{2-0}, dst{4-0} };
}

class V6_vaddbq_enc : Enc_COPROC_VX_p2op<0b01000>;
class V6_vaddhq_enc : Enc_COPROC_VX_p2op<0b01001>;
class V6_vaddwq_enc : Enc_COPROC_VX_p2op<0b01010>;
class V6_vaddbnq_enc : Enc_COPROC_VX_p2op<0b01011>;
class V6_vaddhnq_enc : Enc_COPROC_VX_p2op<0b01100>;
class V6_vaddwnq_enc : Enc_COPROC_VX_p2op<0b01101>;
class V6_vsubbq_enc : Enc_COPROC_VX_p2op<0b01110>;
class V6_vsubhq_enc : Enc_COPROC_VX_p2op<0b01111>;
class V6_vsubwq_enc : Enc_COPROC_VX_p2op<0b10000>;
class V6_vsubbnq_enc : Enc_COPROC_VX_p2op<0b10001>;
class V6_vsubhnq_enc : Enc_COPROC_VX_p2op<0b10010>;
class V6_vsubwnq_enc : Enc_COPROC_VX_p2op<0b10011>;

class Enc_COPROC_VX_2op<bits<6> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<5> src1;

  let Inst{31-16} = { 0b00011110000000, opc{5-4} };
  let Inst{13-0} = { opc{3}, src1{4-0}, opc{2-0}, dst{4-0} };
}

class V6_vabsh_enc : Enc_COPROC_VX_2op<0b000000>;
class V6_vabsh_sat_enc : Enc_COPROC_VX_2op<0b000001>;
class V6_vabsw_enc : Enc_COPROC_VX_2op<0b000010>;
class V6_vabsw_sat_enc : Enc_COPROC_VX_2op<0b000011>;
class V6_vnot_enc : Enc_COPROC_VX_2op<0b000100>;
class V6_vdealh_enc : Enc_COPROC_VX_2op<0b000110>;
class V6_vdealb_enc : Enc_COPROC_VX_2op<0b000111>;
class V6_vunpackob_enc : Enc_COPROC_VX_2op<0b001000>;
class V6_vunpackoh_enc : Enc_COPROC_VX_2op<0b001001>;
class V6_vunpackub_enc : Enc_COPROC_VX_2op<0b010000>;
class V6_vunpackuh_enc : Enc_COPROC_VX_2op<0b010001>;
class V6_vunpackb_enc : Enc_COPROC_VX_2op<0b010010>;
class V6_vunpackh_enc : Enc_COPROC_VX_2op<0b010011>;
class V6_vshuffh_enc : Enc_COPROC_VX_2op<0b010111>;
class V6_vshuffb_enc : Enc_COPROC_VX_2op<0b100000>;
class V6_vzb_enc : Enc_COPROC_VX_2op<0b100001>;
class V6_vzh_enc : Enc_COPROC_VX_2op<0b100010>;
class V6_vsb_enc : Enc_COPROC_VX_2op<0b100011>;
class V6_vsh_enc : Enc_COPROC_VX_2op<0b100100>;
class V6_vcl0w_enc : Enc_COPROC_VX_2op<0b100101>;
class V6_vpopcounth_enc : Enc_COPROC_VX_2op<0b100110>;
class V6_vcl0h_enc : Enc_COPROC_VX_2op<0b100111>;
class V6_vnormamtw_enc : Enc_COPROC_VX_2op<0b110100>;
class V6_vnormamth_enc : Enc_COPROC_VX_2op<0b110101>;
class V6_vassign_enc : Enc_COPROC_VX_2op<0b111111>;

class Enc_COPROC_VMEM_vL32_b_ai<bits<4> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<5> src1;
  bits<10> src2;
  bits<4> src2_vector;

  let src2_vector = src2{9-6};
  let Inst{31-16} = { 0b001010000, opc{3}, 0, src1{4-0} };
  let Inst{13-0} = { src2_vector{3}, 0b00, src2_vector{2-0}, opc{2-0}, dst{4-0} };
}

class V6_vL32b_ai_enc : Enc_COPROC_VMEM_vL32_b_ai<0b0000>;
class V6_vL32b_cur_ai_enc : Enc_COPROC_VMEM_vL32_b_ai<0b0001>;
class V6_vL32b_tmp_ai_enc : Enc_COPROC_VMEM_vL32_b_ai<0b0010>;
class V6_vL32Ub_ai_enc : Enc_COPROC_VMEM_vL32_b_ai<0b0111>;
class V6_vL32b_nt_ai_enc : Enc_COPROC_VMEM_vL32_b_ai<0b1000>;
class V6_vL32b_nt_cur_ai_enc : Enc_COPROC_VMEM_vL32_b_ai<0b1001>;
class V6_vL32b_nt_tmp_ai_enc : Enc_COPROC_VMEM_vL32_b_ai<0b1010>;

class Enc_COPROC_VMEM_vL32_b_ai_128B<bits<4> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<5> src1;
  bits<11> src2;
  bits<4> src2_vector;

  let src2_vector = src2{10-7};
  let Inst{31-16} = { 0b001010000, opc{3}, 0, src1{4-0} };
  let Inst{13-0} = { src2_vector{3}, 0b00, src2_vector{2-0}, opc{2-0}, dst{4-0} };
}

class V6_vL32b_ai_128B_enc : Enc_COPROC_VMEM_vL32_b_ai_128B<0b0000>;
class V6_vL32b_cur_ai_128B_enc : Enc_COPROC_VMEM_vL32_b_ai_128B<0b0001>;
class V6_vL32b_tmp_ai_128B_enc : Enc_COPROC_VMEM_vL32_b_ai_128B<0b0010>;
class V6_vL32Ub_ai_128B_enc : Enc_COPROC_VMEM_vL32_b_ai_128B<0b0111>;
class V6_vL32b_nt_ai_128B_enc : Enc_COPROC_VMEM_vL32_b_ai_128B<0b1000>;
class V6_vL32b_nt_cur_ai_128B_enc : Enc_COPROC_VMEM_vL32_b_ai_128B<0b1001>;
class V6_vL32b_nt_tmp_ai_128B_enc : Enc_COPROC_VMEM_vL32_b_ai_128B<0b1010>;

class Enc_COPROC_VMEM_vS32_b_ai_64B<bits<4> opc> : OpcodeHexagon {
  bits<5> src1;
  bits<10> src2;
  bits<4> src2_vector;
  bits<5> src3;

  let src2_vector = src2{9-6};
  let Inst{31-16} = { 0b001010000, opc{3}, 1, src1{4-0} };
  let Inst{13-0} = { src2_vector{3}, 0b00, src2_vector{2-0}, opc{2-0}, src3{4-0} };
}

class Enc_COPROC_VMEM_vS32_b_ai_128B<bits<4> opc> : OpcodeHexagon {
  bits<5> src1;
  bits<11> src2;
  bits<4> src2_vector;
  bits<5> src3;

  let src2_vector = src2{10-7};
  let Inst{31-16} = { 0b001010000, opc{3}, 1, src1{4-0} };
  let Inst{13-0} = { src2_vector{3}, 0b00, src2_vector{2-0}, opc{2-0}, src3{4-0} };
}

class V6_vS32b_ai_enc : Enc_COPROC_VMEM_vS32_b_ai_64B<0b0000>;
class V6_vS32Ub_ai_enc : Enc_COPROC_VMEM_vS32_b_ai_64B<0b0111>;
class V6_vS32b_nt_ai_enc : Enc_COPROC_VMEM_vS32_b_ai_64B<0b1000>;

class V6_vS32b_ai_128B_enc : Enc_COPROC_VMEM_vS32_b_ai_128B<0b0000>;
class V6_vS32Ub_ai_128B_enc : Enc_COPROC_VMEM_vS32_b_ai_128B<0b0111>;
class V6_vS32b_nt_ai_128B_enc : Enc_COPROC_VMEM_vS32_b_ai_128B<0b1000>;

class Enc_COPROC_VMEM_vS32b_n_ew_ai_64B<bits<1> opc> : OpcodeHexagon {
  bits<5> src1;
  bits<10> src2;
  bits<4> src2_vector;
  bits<3> src3;

  let src2_vector = src2{9-6};
  let Inst{31-16} = { 0b001010000, opc{0}, 1, src1{4-0} };
  let Inst{13-0} = { src2_vector{3}, 0b00, src2_vector{2-0}, 0b00100, src3{2-0} };
}

class V6_vS32b_new_ai_enc : Enc_COPROC_VMEM_vS32b_n_ew_ai_64B<0>;
class V6_vS32b_nt_new_ai_enc : Enc_COPROC_VMEM_vS32b_n_ew_ai_64B<1>;

class Enc_COPROC_VMEM_vS32b_n_ew_ai_128B<bits<1> opc> : OpcodeHexagon {
  bits<5> src1;
  bits<11> src2;
  bits<4> src2_vector;
  bits<3> src3;

  let src2_vector = src2{10-7};
  let Inst{31-16} = { 0b001010000, opc{0}, 1, src1{4-0} };
  let Inst{13-0} = { src2_vector{3}, 0b00, src2_vector{2-0}, 0b00100, src3{2-0} };
}

class V6_vS32b_new_ai_128B_enc : Enc_COPROC_VMEM_vS32b_n_ew_ai_128B<0>;
class V6_vS32b_nt_new_ai_128B_enc : Enc_COPROC_VMEM_vS32b_n_ew_ai_128B<1>;

class Enc_COPROC_VMEM_vS32_b_pred_ai<bits<5> opc> : OpcodeHexagon {
  bits<2> src1;
  bits<5> src2;
  bits<10> src3;
  bits<4> src3_vector;
  bits<5> src4;

  let src3_vector = src3{9-6};
  let Inst{31-16} = { 0b001010001, opc{4-3}, src2{4-0} };
  let Inst{13-0} = { src3_vector{3}, src1{1-0}, src3_vector{2-0}, opc{2-0}, src4{4-0} };
}

class Enc_COPROC_VMEM_vS32_b_pred_ai_128B<bits<5> opc> : OpcodeHexagon {
  bits<2> src1;
  bits<5> src2;
  bits<11> src3;
  bits<4> src3_vector;
  bits<5> src4;

  let src3_vector = src3{10-7};
  let Inst{31-16} = { 0b001010001, opc{4-3}, src2{4-0} };
  let Inst{13-0} = { src3_vector{3}, src1{1-0}, src3_vector{2-0}, opc{2-0}, src4{4-0} };
}

class V6_vS32b_qpred_ai_enc : Enc_COPROC_VMEM_vS32_b_pred_ai<0b00000>;
class V6_vS32b_nqpred_ai_enc : Enc_COPROC_VMEM_vS32_b_pred_ai<0b00001>;
class V6_vS32b_pred_ai_enc : Enc_COPROC_VMEM_vS32_b_pred_ai<0b01000>;
class V6_vS32b_npred_ai_enc : Enc_COPROC_VMEM_vS32_b_pred_ai<0b01001>;
class V6_vS32Ub_pred_ai_enc : Enc_COPROC_VMEM_vS32_b_pred_ai<0b01110>;
class V6_vS32Ub_npred_ai_enc : Enc_COPROC_VMEM_vS32_b_pred_ai<0b01111>;
class V6_vS32b_nt_qpred_ai_enc : Enc_COPROC_VMEM_vS32_b_pred_ai<0b10000>;
class V6_vS32b_nt_nqpred_ai_enc : Enc_COPROC_VMEM_vS32_b_pred_ai<0b10001>;
class V6_vS32b_nt_pred_ai_enc : Enc_COPROC_VMEM_vS32_b_pred_ai<0b11000>;
class V6_vS32b_nt_npred_ai_enc : Enc_COPROC_VMEM_vS32_b_pred_ai<0b11001>;

class V6_vS32b_qpred_ai_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_ai_128B<0b00000>;
class V6_vS32b_nqpred_ai_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_ai_128B<0b00001>;
class V6_vS32b_pred_ai_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_ai_128B<0b01000>;
class V6_vS32b_npred_ai_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_ai_128B<0b01001>;
class V6_vS32Ub_pred_ai_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_ai_128B<0b01110>;
class V6_vS32Ub_npred_ai_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_ai_128B<0b01111>;
class V6_vS32b_nt_qpred_ai_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_ai_128B<0b10000>;
class V6_vS32b_nt_nqpred_ai_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_ai_128B<0b10001>;
class V6_vS32b_nt_pred_ai_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_ai_128B<0b11000>;
class V6_vS32b_nt_npred_ai_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_ai_128B<0b11001>;

class Enc_COPROC_VMEM_vS32b_n_ew_pred_ai<bits<4> opc> : OpcodeHexagon {
  bits<2> src1;
  bits<5> src2;
  bits<10> src3;
  bits<4> src3_vector;
  bits<3> src4;

  let src3_vector = src3{9-6};
  let Inst{31-16} = { 0b001010001, opc{3}, 1, src2{4-0} };
  let Inst{13-0} = { src3_vector{3}, src1{1-0}, src3_vector{2-0}, 0b01, opc{2-0}, src4{2-0} };
}

class V6_vS32b_new_pred_ai_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_ai<0b0000>;
class V6_vS32b_new_npred_ai_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_ai<0b0101>;
class V6_vS32b_nt_new_pred_ai_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_ai<0b1010>;
class V6_vS32b_nt_new_npred_ai_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_ai<0b1111>;

class Enc_COPROC_VMEM_vS32b_n_ew_pred_ai_128B<bits<4> opc> : OpcodeHexagon {
  bits<2> src1;
  bits<5> src2;
  bits<11> src3;
  bits<4> src3_vector;
  bits<3> src4;

  let src3_vector = src3{10-7};
  let Inst{31-16} = { 0b001010001, opc{3}, 1, src2{4-0} };
  let Inst{13-0} = { src3_vector{3}, src1{1-0}, src3_vector{2-0}, 0b01, opc{2-0}, src4{2-0} };
}

class V6_vS32b_new_pred_ai_128B_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_ai_128B<0b0000>;
class V6_vS32b_new_npred_ai_128B_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_ai_128B<0b0101>;
class V6_vS32b_nt_new_pred_ai_128B_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_ai_128B<0b1010>;
class V6_vS32b_nt_new_npred_ai_128B_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_ai_128B<0b1111>;

// TODO: Change script to generate dst, src1, src2 instead of
// dst, dst2, src1.
class Enc_COPROC_VMEM_vL32_b_pi<bits<4> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<5> src1;
  bits<9> src2;
  bits<3> src2_vector;

  let src2_vector = src2{8-6};
  let Inst{31-16} = { 0b001010010, opc{3}, 0, src1{4-0} };
  let Inst{13-0} = { 0b000, src2_vector{2-0}, opc{2-0}, dst{4-0} };
}

class V6_vL32b_pi_enc : Enc_COPROC_VMEM_vL32_b_pi<0b0000>;
class V6_vL32b_cur_pi_enc : Enc_COPROC_VMEM_vL32_b_pi<0b0001>;
class V6_vL32b_tmp_pi_enc : Enc_COPROC_VMEM_vL32_b_pi<0b0010>;
class V6_vL32Ub_pi_enc : Enc_COPROC_VMEM_vL32_b_pi<0b0111>;
class V6_vL32b_nt_pi_enc : Enc_COPROC_VMEM_vL32_b_pi<0b1000>;
class V6_vL32b_nt_cur_pi_enc : Enc_COPROC_VMEM_vL32_b_pi<0b1001>;
class V6_vL32b_nt_tmp_pi_enc : Enc_COPROC_VMEM_vL32_b_pi<0b1010>;

class Enc_COPROC_VMEM_vL32_b_pi_128B<bits<4> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<5> src1;
  bits<10> src2;
  bits<3> src2_vector;

  let src2_vector = src2{9-7};
  let Inst{31-16} = { 0b001010010, opc{3}, 0, src1{4-0} };
  let Inst{13-0} = { 0b000, src2_vector{2-0}, opc{2-0}, dst{4-0} };
}

class V6_vL32b_pi_128B_enc : Enc_COPROC_VMEM_vL32_b_pi_128B<0b0000>;
class V6_vL32b_cur_pi_128B_enc : Enc_COPROC_VMEM_vL32_b_pi_128B<0b0001>;
class V6_vL32b_tmp_pi_128B_enc : Enc_COPROC_VMEM_vL32_b_pi_128B<0b0010>;
class V6_vL32Ub_pi_128B_enc : Enc_COPROC_VMEM_vL32_b_pi_128B<0b0111>;
class V6_vL32b_nt_pi_128B_enc : Enc_COPROC_VMEM_vL32_b_pi_128B<0b1000>;
class V6_vL32b_nt_cur_pi_128B_enc : Enc_COPROC_VMEM_vL32_b_pi_128B<0b1001>;
class V6_vL32b_nt_tmp_pi_128B_enc : Enc_COPROC_VMEM_vL32_b_pi_128B<0b1010>;


// TODO: Change script to generate src1, src2 and src3 instead of
// dst, src1, src2.
class Enc_COPROC_VMEM_vS32_b_pi<bits<4> opc> : OpcodeHexagon {
  bits<5> src1;
  bits<9> src2;
  bits<3> src2_vector;
  bits<5> src3;

  let src2_vector = src2{8-6};
  let Inst{31-16} = { 0b001010010, opc{3}, 1, src1{4-0} };
  let Inst{10-0} = {src2_vector{2-0}, opc{2-0}, src3{4-0} };
}

class V6_vS32b_pi_enc : Enc_COPROC_VMEM_vS32_b_pi<0b0000>;
class V6_vS32Ub_pi_enc : Enc_COPROC_VMEM_vS32_b_pi<0b0111>;
class V6_vS32b_nt_pi_enc : Enc_COPROC_VMEM_vS32_b_pi<0b1000>;

class Enc_COPROC_VMEM_vS32_b_pi_128B<bits<4> opc> : OpcodeHexagon {
  bits<5> src1;
  bits<10> src2;
  bits<3> src2_vector;
  bits<5> src3;

  let src2_vector = src2{9-7};
  let Inst{31-16} = { 0b001010010, opc{3}, 1, src1{4-0} };
  let Inst{10-0} = {src2_vector{2-0}, opc{2-0}, src3{4-0} };
}

class V6_vS32b_pi_128B_enc : Enc_COPROC_VMEM_vS32_b_pi_128B<0b0000>;
class V6_vS32Ub_pi_128B_enc : Enc_COPROC_VMEM_vS32_b_pi_128B<0b0111>;
class V6_vS32b_nt_pi_128B_enc : Enc_COPROC_VMEM_vS32_b_pi_128B<0b1000>;

// TODO: Change script to generate src1, src2 and src3 instead of
// dst, src1, src2.
class Enc_COPROC_VMEM_vS32b_n_ew_pi<bits<1> opc> : OpcodeHexagon {
  bits<5> src1;
  bits<9> src2;
  bits<3> src2_vector;
  bits<3> src3;

  let src2_vector = src2{8-6};
  let Inst{31-16} = { 0b001010010, opc{0}, 1, src1{4-0} };
  let Inst{13-0} = { 0b000, src2_vector{2-0}, 0b00100, src3{2-0} };
}

class V6_vS32b_new_pi_enc : Enc_COPROC_VMEM_vS32b_n_ew_pi<0>;
class V6_vS32b_nt_new_pi_enc : Enc_COPROC_VMEM_vS32b_n_ew_pi<1>;

class Enc_COPROC_VMEM_vS32b_n_ew_pi_128B<bits<1> opc> : OpcodeHexagon {
  bits<5> src1;
  bits<10> src2;
  bits<3> src2_vector;
  bits<3> src3;

  let src2_vector = src2{9-7};
  let Inst{31-16} = { 0b001010010, opc{0}, 1, src1{4-0} };
  let Inst{13-0} = { 0b000, src2_vector{2-0}, 0b00100, src3{2-0} };
}

class V6_vS32b_new_pi_128B_enc : Enc_COPROC_VMEM_vS32b_n_ew_pi_128B<0>;
class V6_vS32b_nt_new_pi_128B_enc : Enc_COPROC_VMEM_vS32b_n_ew_pi_128B<1>;

// TODO: Change script to generate src1, src2,src3 and src4 instead of
// dst, src1, src2, src3.
class Enc_COPROC_VMEM_vS32_b_pred_pi<bits<5> opc> : OpcodeHexagon {
  bits<2> src1;
  bits<5> src2;
  bits<9> src3;
  bits<3> src3_vector;
  bits<5> src4;

  let src3_vector = src3{8-6};
  let Inst{31-16} = { 0b001010011, opc{4-3}, src2{4-0} };
  let Inst{13-0} = { 0, src1{1-0}, src3_vector{2-0}, opc{2-0}, src4{4-0} };
}

class V6_vS32b_qpred_pi_enc : Enc_COPROC_VMEM_vS32_b_pred_pi<0b00000>;
class V6_vS32b_nqpred_pi_enc : Enc_COPROC_VMEM_vS32_b_pred_pi<0b00001>;
class V6_vS32b_pred_pi_enc : Enc_COPROC_VMEM_vS32_b_pred_pi<0b01000>;
class V6_vS32b_npred_pi_enc : Enc_COPROC_VMEM_vS32_b_pred_pi<0b01001>;
class V6_vS32Ub_pred_pi_enc : Enc_COPROC_VMEM_vS32_b_pred_pi<0b01110>;
class V6_vS32Ub_npred_pi_enc : Enc_COPROC_VMEM_vS32_b_pred_pi<0b01111>;
class V6_vS32b_nt_qpred_pi_enc : Enc_COPROC_VMEM_vS32_b_pred_pi<0b10000>;
class V6_vS32b_nt_nqpred_pi_enc : Enc_COPROC_VMEM_vS32_b_pred_pi<0b10001>;
class V6_vS32b_nt_pred_pi_enc : Enc_COPROC_VMEM_vS32_b_pred_pi<0b11000>;
class V6_vS32b_nt_npred_pi_enc : Enc_COPROC_VMEM_vS32_b_pred_pi<0b11001>;

// TODO: Change script to generate src1, src2,src3 and src4 instead of
// dst, src1, src2, src3.
class Enc_COPROC_VMEM_vS32_b_pred_pi_128B<bits<5> opc> : OpcodeHexagon {
  bits<2> src1;
  bits<5> src2;
  bits<10> src3;
  bits<3> src3_vector;
  bits<5> src4;

  let src3_vector = src3{9-7};
  let Inst{31-16} = { 0b001010011, opc{4-3}, src2{4-0} };
  let Inst{13-0} = { 0, src1{1-0}, src3_vector{2-0}, opc{2-0}, src4{4-0} };
}

class V6_vS32b_qpred_pi_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_pi_128B<0b00000>;
class V6_vS32b_nqpred_pi_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_pi_128B<0b00001>;
class V6_vS32b_pred_pi_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_pi_128B<0b01000>;
class V6_vS32b_npred_pi_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_pi_128B<0b01001>;
class V6_vS32Ub_pred_pi_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_pi_128B<0b01110>;
class V6_vS32Ub_npred_pi_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_pi_128B<0b01111>;
class V6_vS32b_nt_qpred_pi_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_pi_128B<0b10000>;
class V6_vS32b_nt_nqpred_pi_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_pi_128B<0b10001>;
class V6_vS32b_nt_pred_pi_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_pi_128B<0b11000>;
class V6_vS32b_nt_npred_pi_128B_enc : Enc_COPROC_VMEM_vS32_b_pred_pi_128B<0b11001>;

class Enc_COPROC_VMEM_vS32b_n_ew_pred_pi<bits<4> opc> : OpcodeHexagon {
  bits<2> src1;
  bits<5> src2;
  bits<9> src3;
  bits<3> src3_vector;
  bits<3> src4;

  let src3_vector = src3{8-6};
  let Inst{31-16} = { 0b001010011, opc{3}, 1, src2{4-0} };
  let Inst{13-0} = { 0, src1{1-0}, src3_vector{2-0}, 0b01, opc{2-0}, src4{2-0} };
}

class V6_vS32b_new_pred_pi_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_pi<0b0000>;
class V6_vS32b_new_npred_pi_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_pi<0b0101>;
class V6_vS32b_nt_new_pred_pi_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_pi<0b1010>;
class V6_vS32b_nt_new_npred_pi_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_pi<0b1111>;

class Enc_COPROC_VMEM_vS32b_n_ew_pred_pi_128B<bits<4> opc> : OpcodeHexagon {
  bits<2> src1;
  bits<5> src2;
  bits<10> src3;
  bits<3> src3_vector;
  bits<3> src4;

  let src3_vector = src3{9-7};
  let Inst{31-16} = { 0b001010011, opc{3}, 1, src2{4-0} };
  let Inst{13-0} = { 0, src1{1-0}, src3_vector{2-0}, 0b01, opc{2-0}, src4{2-0} };
}

class V6_vS32b_new_pred_pi_128B_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_pi_128B<0b0000>;
class V6_vS32b_new_npred_pi_128B_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_pi_128B<0b0101>;
class V6_vS32b_nt_new_pred_pi_128B_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_pi_128B<0b1010>;
class V6_vS32b_nt_new_npred_pi_128B_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_pi_128B<0b1111>;

class Enc_LD_load_m<bits<13> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<5> src1;
  bits<1> src2;

  let Inst{31-16} = { opc{12}, 0, opc{11-10}, 1, opc{9-4}, src1{4-0} };
  let Inst{13-0} = { src2{0}, 0b000, opc{3}, 0, opc{2-0}, dst{4-0} };
}

class V6_vL32b_ppu_enc : Enc_LD_load_m<0b0100110000000>;
class V6_vL32b_cur_ppu_enc : Enc_LD_load_m<0b0100110000001>;
class V6_vL32b_tmp_ppu_enc : Enc_LD_load_m<0b0100110000010>;
class V6_vL32Ub_ppu_enc : Enc_LD_load_m<0b0100110000111>;
class V6_vL32b_nt_ppu_enc : Enc_LD_load_m<0b0100110100000>;
class V6_vL32b_nt_cur_ppu_enc : Enc_LD_load_m<0b0100110100001>;
class V6_vL32b_nt_tmp_ppu_enc : Enc_LD_load_m<0b0100110100010>;

class Enc_COPROC_VMEM_vS32_b_ppu<bits<4> opc> : OpcodeHexagon {
  bits<5> src1;
  bits<1> src2;
  bits<5> src3;

  let Inst{31-16} = { 0b001010110, opc{3}, 1, src1{4-0} };
  let Inst{13-0} = { src2{0}, 0b00000, opc{2-0}, src3{4-0} };
}

class V6_vS32b_ppu_enc : Enc_COPROC_VMEM_vS32_b_ppu<0b0000>;
class V6_vS32Ub_ppu_enc : Enc_COPROC_VMEM_vS32_b_ppu<0b0111>;
class V6_vS32b_nt_ppu_enc : Enc_COPROC_VMEM_vS32_b_ppu<0b1000>;

class Enc_COPROC_VMEM_vS32b_new_ppu<bits<1> opc> : OpcodeHexagon {
  bits<5> src1;
  bits<1> src2;
  bits<3> src3;

  let Inst{31-16} = { 0b001010110, opc{0}, 1, src1{4-0} };
  let Inst{13-0} = { src2{0}, 0b0000000100, src3{2-0} };
}

class V6_vS32b_new_ppu_enc : Enc_COPROC_VMEM_vS32b_new_ppu<0>;
class V6_vS32b_nt_new_ppu_enc : Enc_COPROC_VMEM_vS32b_new_ppu<1>;

class Enc_COPROC_VMEM_vS32_b_pred_ppu<bits<5> opc> : OpcodeHexagon {
  bits<2> src1;
  bits<5> src2;
  bits<1> src3;
  bits<5> src4;

  let Inst{31-16} = { 0b001010111, opc{4-3}, src2{4-0} };
  let Inst{13-0} = { src3{0}, src1{1-0}, 0b000, opc{2-0}, src4{4-0} };
}

class V6_vS32b_qpred_ppu_enc : Enc_COPROC_VMEM_vS32_b_pred_ppu<0b00000>;
class V6_vS32b_nqpred_ppu_enc : Enc_COPROC_VMEM_vS32_b_pred_ppu<0b00001>;
class V6_vS32b_pred_ppu_enc : Enc_COPROC_VMEM_vS32_b_pred_ppu<0b01000>;
class V6_vS32b_npred_ppu_enc : Enc_COPROC_VMEM_vS32_b_pred_ppu<0b01001>;
class V6_vS32Ub_pred_ppu_enc : Enc_COPROC_VMEM_vS32_b_pred_ppu<0b01110>;
class V6_vS32Ub_npred_ppu_enc : Enc_COPROC_VMEM_vS32_b_pred_ppu<0b01111>;
class V6_vS32b_nt_qpred_ppu_enc : Enc_COPROC_VMEM_vS32_b_pred_ppu<0b10000>;
class V6_vS32b_nt_nqpred_ppu_enc : Enc_COPROC_VMEM_vS32_b_pred_ppu<0b10001>;
class V6_vS32b_nt_pred_ppu_enc : Enc_COPROC_VMEM_vS32_b_pred_ppu<0b11000>;
class V6_vS32b_nt_npred_ppu_enc : Enc_COPROC_VMEM_vS32_b_pred_ppu<0b11001>;

class Enc_COPROC_VMEM_vS32b_n_ew_pred_ppu<bits<4> opc> : OpcodeHexagon {
  bits<2> src1;
  bits<5> src2;
  bits<1> src3;
  bits<3> src4;

  let Inst{31-16} = { 0b001010111, opc{3}, 1, src2{4-0} };
  let Inst{13-0} = { src3{0}, src1{1-0}, 0b00001, opc{2-0}, src4{2-0} };
}

class V6_vS32b_new_pred_ppu_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_ppu<0b0000>;
class V6_vS32b_new_npred_ppu_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_ppu<0b0101>;
class V6_vS32b_nt_new_pred_ppu_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_ppu<0b1010>;
class V6_vS32b_nt_new_npred_ppu_enc : Enc_COPROC_VMEM_vS32b_n_ew_pred_ppu<0b1111>;


class Enc_COPROC_VX_4op_i<bits<5> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<5> src1;
  bits<5> src2;
  bits<1> src3;

  let Inst{31-16} = { 0b00011001, opc{4-2}, src2{4-0} };
  let Inst{13-0} = { opc{1}, src1{4-0}, 1, opc{0}, src3{0}, dst{4-0} };
}

class V6_vrmpybusi_enc : Enc_COPROC_VX_4op_i<0b01000>;
class V6_vrsadubi_enc : Enc_COPROC_VX_4op_i<0b01001>;
class V6_vrmpybusi_acc_enc : Enc_COPROC_VX_4op_i<0b01010>;
class V6_vrsadubi_acc_enc : Enc_COPROC_VX_4op_i<0b01011>;
class V6_vrmpyubi_acc_enc : Enc_COPROC_VX_4op_i<0b01111>;
class V6_vrmpyubi_enc : Enc_COPROC_VX_4op_i<0b10101>;

class Enc_COPROC_VX_vandqrt<bits<5> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<2> src1;
  bits<5> src2;

  let Inst{31-16} = { 0b00011001, opc{4-3}, 1, src2{4-0} };
  let Inst{13-0} = { opc{2}, 0b000, src1{1-0}, opc{1-0}, 1, dst{4-0} };
}

class V6_vandqrt_acc_enc : Enc_COPROC_VX_vandqrt<0b01101>;
class V6_vandqrt_enc : Enc_COPROC_VX_vandqrt<0b10010>;

class Enc_COPROC_VX_cards<bits<2> opc> : OpcodeHexagon {
  bits<5> src1;
  bits<5> src2;
  bits<5> src3;

  let Inst{31-16} = { 0b00011001111, src3{4-0} };
  let Inst{13-0} = { 1, src1{4-0}, 0, opc{1-0}, src2{4-0} };
}

class V6_vshuff_enc : Enc_COPROC_VX_cards<0b01>;
class V6_vdeal_enc : Enc_COPROC_VX_cards<0b10>;


class Enc_COPROC_VX_v_cmov<bits<1> opc> : OpcodeHexagon {
  bits<2> src1;
  bits<5> dst;
  bits<5> src2;

  let Inst{31-16} = { 0b0001101000, opc{0}, 0b00000 };
  let Inst{13-0} = { 0, src2{4-0}, 0, src1{1-0}, dst{4-0} };
}

class V6_vcmov_enc : Enc_COPROC_VX_v_cmov<0>;
class V6_vncmov_enc : Enc_COPROC_VX_v_cmov<1>;

class Enc_X_p3op<bits<8> opc> : OpcodeHexagon {
  bits<2> src1;
  bits<5> dst;
  bits<5> src2;
  bits<5> src3;

  let Inst{31-16} = { opc{7-5}, 0b1101, opc{4}, 0, opc{3-2}, src3{4-0} };
  let Inst{13-0} = { opc{1}, src2{4-0}, opc{0}, src1{1-0}, dst{4-0} };
}

class V6_vnccombine_enc : Enc_X_p3op<0b00001000>;
class V6_vccombine_enc : Enc_X_p3op<0b00001100>;

class Enc_COPROC_VX_4op_r<bits<4> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<5> src1;
  bits<5> src2;
  bits<3> src3;

  let Inst{31-16} = { 0b00011011, src2{4-0}, src3{2-0} };
  let Inst{13-0} = { opc{3}, src1{4-0}, opc{2-0}, dst{4-0} };
}

class V6_valignb_enc : Enc_COPROC_VX_4op_r<0b0000>;
class V6_vlalignb_enc : Enc_COPROC_VX_4op_r<0b0001>;
class V6_vasrwh_enc : Enc_COPROC_VX_4op_r<0b0010>;
class V6_vasrwhsat_enc : Enc_COPROC_VX_4op_r<0b0011>;
class V6_vasrwhrndsat_enc : Enc_COPROC_VX_4op_r<0b0100>;
class V6_vasrwuhsat_enc : Enc_COPROC_VX_4op_r<0b0101>;
class V6_vasrhubsat_enc : Enc_COPROC_VX_4op_r<0b0110>;
class V6_vasrhubrndsat_enc : Enc_COPROC_VX_4op_r<0b0111>;
class V6_vasrhbrndsat_enc : Enc_COPROC_VX_4op_r<0b1000>;
class V6_vlutvvb_enc : Enc_COPROC_VX_4op_r<0b1001>;
class V6_vshuffvdd_enc : Enc_COPROC_VX_4op_r<0b1011>;
class V6_vdealvdd_enc : Enc_COPROC_VX_4op_r<0b1100>;
class V6_vlutvvb_oracc_enc : Enc_COPROC_VX_4op_r<0b1101>;
class V6_vlutvwh_enc : Enc_COPROC_VX_4op_r<0b1110>;
class V6_vlutvwh_oracc_enc : Enc_COPROC_VX_4op_r<0b1111>;

class Enc_S_3op_valign_i<bits<9> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<5> src1;
  bits<5> src2;
  bits<3> src3;

  let Inst{31-16} = { opc{8-7}, 0, opc{6-3}, 0b00, opc{2-1}, src2{4-0} };
  let Inst{13-0} = { opc{0}, src1{4-0}, src3{2-0}, dst{4-0} };
}

class V6_vlutb_enc : Enc_S_3op_valign_i<0b001100000>;
class V6_vlutb_dv_enc : Enc_S_3op_valign_i<0b001100010>;
class V6_vlutb_acc_enc : Enc_S_3op_valign_i<0b001100100>;
class V6_vlutb_dv_acc_enc : Enc_S_3op_valign_i<0b001100110>;
class V6_valignbi_enc : Enc_S_3op_valign_i<0b001111011>;
class V6_vlalignbi_enc : Enc_S_3op_valign_i<0b001111111>;
class S2_valignib_enc : Enc_S_3op_valign_i<0b110000000>;
class S2_addasl_rrri_enc : Enc_S_3op_valign_i<0b110010000>;

class Enc_COPROC_VX_3op_q<bits<3> opc> : OpcodeHexagon {
  bits<2> dst;
  bits<2> src1;
  bits<2> src2;

  let Inst{31-16} = { 0b00011110, src2{1-0}, 0b000011 };
  let Inst{13-0} = { 0b0000, src1{1-0}, 0b000, opc{2-0}, dst{1-0} };
}

class V6_pred_and_enc : Enc_COPROC_VX_3op_q<0b000>;
class V6_pred_or_enc : Enc_COPROC_VX_3op_q<0b001>;
class V6_pred_xor_enc : Enc_COPROC_VX_3op_q<0b011>;
class V6_pred_or_n_enc : Enc_COPROC_VX_3op_q<0b100>;
class V6_pred_and_n_enc : Enc_COPROC_VX_3op_q<0b101>;

class V6_pred_not_enc : OpcodeHexagon {
  bits<2> dst;
  bits<2> src1;

  let Inst{31-16} = { 0b0001111000000011 };
  let Inst{13-0} = { 0b0000, src1{1-0}, 0b000010, dst{1-0} };
}

class Enc_COPROC_VX_4op_q<bits<1> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<2> src1;
  bits<5> src2;
  bits<5> src3;

  let Inst{31-16} = { 0b000111101, opc{0}, 1, src3{4-0} };
  let Inst{13-0} = { 1, src2{4-0}, 0, src1{1-0}, dst{4-0} };
}

class V6_vswap_enc : Enc_COPROC_VX_4op_q<0>;
class V6_vmux_enc : Enc_COPROC_VX_4op_q<1>;

class Enc_X_2op<bits<16> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<5> src1;

  let Inst{31-16} = { opc{15-5}, src1{4-0} };
  let Inst{13-0} = { opc{4-3}, 0b0000, opc{2-0}, dst{4-0} };
}

class V6_lvsplatw_enc : Enc_X_2op<0b0001100110100001>;
class V6_vinsertwr_enc : Enc_X_2op<0b0001100110110001>;
class S6_vsplatrbp_enc : Enc_X_2op<0b1000010001000100>;


class Enc_CR_2op_r<bits<12> opc> : OpcodeHexagon {
  bits<2> dst;
  bits<5> src1;

  let Inst{31-16} = { opc{11}, 0, opc{10-7}, 0, opc{6-3}, src1{4-0} };
  let Inst{13-0} = { opc{2}, 0b000000, opc{1}, 0b000, opc{0}, dst{1-0} };
}

class V6_pred_scalar2_enc : Enc_CR_2op_r<0b001101101011>;
class Y5_l2locka_enc : Enc_CR_2op_r<0b110000111100>;

class Enc_S_3op_i6<bits<9> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<5> src1;
  bits<6> src2;

  let Inst{31-16} = { 0b1000, opc{8-6}, 0, opc{5-3}, src1{4-0} };
  let Inst{13-0} = { src2{5-0}, opc{2-0}, dst{4-0} };
}

class S6_rol_i_p_enc : Enc_S_3op_i6<0b000000011>;
class S6_rol_i_p_nac_enc : Enc_S_3op_i6<0b001000011>;
class S6_rol_i_p_acc_enc : Enc_S_3op_i6<0b001000111>;
class S6_rol_i_p_and_enc : Enc_S_3op_i6<0b001010011>;
class S6_rol_i_p_or_enc : Enc_S_3op_i6<0b001010111>;
class S6_rol_i_p_xacc_enc : Enc_S_3op_i6<0b001100011>;

class Enc_X_3op_r<bits<15> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<5> src1;
  bits<5> src2;

  let Inst{31-16} = { opc{14-4}, src1{4-0} };
  let Inst{13-0} = { opc{3}, src2{4-0}, opc{2-0}, dst{4-0} };
}

class S6_rol_i_r_enc : Enc_X_3op_r<0b100011000000011>;
class S6_rol_i_r_nac_enc : Enc_X_3op_r<0b100011100000011>;
class S6_rol_i_r_acc_enc : Enc_X_3op_r<0b100011100000111>;
class S6_rol_i_r_and_enc : Enc_X_3op_r<0b100011100100011>;
class S6_rol_i_r_or_enc : Enc_X_3op_r<0b100011100100111>;
class S6_rol_i_r_xacc_enc : Enc_X_3op_r<0b100011101000011>;
class S6_vtrunehb_ppp_enc : Enc_X_3op_r<0b110000011000011>;
class S6_vtrunohb_ppp_enc : Enc_X_3op_r<0b110000011000101>;

class Enc_no_operands<bits<25> opc> : OpcodeHexagon {

  let Inst{31-16} = { opc{24-10}, 0 };
  let Inst{13-0} = { opc{9-7}, 0b000, opc{6-0}, 0 };
}

class Y5_l2gunlock_enc : Enc_no_operands<0b1010100000100000010000000>;
class Y5_l2gclean_enc : Enc_no_operands<0b1010100000100000100000000>;
class Y5_l2gcleaninv_enc : Enc_no_operands<0b1010100000100000110000000>;
class V6_vhist_enc : Enc_no_operands<0b0001111000000001001000000>;

class Enc_J_jumpr<bits<13> opc> : OpcodeHexagon {
  bits<5> src1;

  let Inst{31-16} = { opc{12-6}, 0, opc{5-3}, src1{4-0} };
  let Inst{13-0} = { 0b00, opc{2}, 0b0000, opc{1-0}, 0b00000 };
}

class Y5_l2unlocka_enc : Enc_J_jumpr<0b1010011011000>;
class Y2_l2cleaninvidx_enc : Enc_J_jumpr<0b1010100011000>;

class Enc_ST_l2gclean_pa<bits<2> opc> : OpcodeHexagon {
  bits<5> src1;

  let Inst{31-16} = { 0b101001101, opc{1-0}, 0b00000 };
  let Inst{13-0} = { 0, src1{4-0}, 0b00000000 };
}

class Y6_l2gcleanpa_enc : Enc_ST_l2gclean_pa<0b01>;
class Y6_l2gcleaninvpa_enc : Enc_ST_l2gclean_pa<0b10>;

class A5_ACS_enc : OpcodeHexagon {
  bits<5> dst1;
  bits<2> dst2;
  bits<5> src1;
  bits<5> src2;

  let Inst{31-16} = { 0b11101010101, src1{4-0} };
  let Inst{13-0} = { 0, src2{4-0}, 0, dst2{1-0}, dst1{4-0} };
}

class Enc_X_4op_r<bits<8> opc> : OpcodeHexagon {
  bits<5> dst;
  bits<5> src1;
  bits<5> src2;
  bits<2> src3;

  let Inst{31-16} = { 0b11, opc{7}, 0, opc{6-5}, 1, opc{4-1}, src1{4-0} };
  let Inst{13-0} = { 0, src2{4-0}, opc{0}, src3{1-0}, dst{4-0} };
}

class S2_vsplicerb_enc : Enc_X_4op_r<0b00001000>;
class S2_cabacencbin_enc : Enc_X_4op_r<0b00001010>;
class F2_sffma_sc_enc : Enc_X_4op_r<0b11110111>;

class V6_vhistq_enc : OpcodeHexagon {
  bits<2> src1;

  let Inst{31-16} = { 0b00011110, src1{1-0}, 0b000010 };
  let Inst{13-0} = { 0b10000010000000 };
}

// TODO: Change script to generate dst1 instead of dst.
class A6_vminub_RdP_enc : OpcodeHexagon {
  bits<5> dst1;
  bits<2> dst2;
  bits<5> src1;
  bits<5> src2;

  let Inst{31-16} = { 0b11101010111, src2{4-0} };
  let Inst{13-0} = { 0, src1{4-0}, 0, dst2{1-0}, dst1{4-0} };
}
