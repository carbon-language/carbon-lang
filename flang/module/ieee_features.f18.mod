!mod$ v1 sum:0fa84ac68ab883f5
module ieee_features
type::ieee_features_type
integer(1),private::feature=0_4
end type
type(ieee_features_type),parameter::ieee_datatype=ieee_features_type(feature=1_1)
type(ieee_features_type),parameter::ieee_denormal=ieee_features_type(feature=2_1)
type(ieee_features_type),parameter::ieee_divide=ieee_features_type(feature=3_1)
type(ieee_features_type),parameter::ieee_halting=ieee_features_type(feature=4_1)
type(ieee_features_type),parameter::ieee_inexact_flag=ieee_features_type(feature=5_1)
type(ieee_features_type),parameter::ieee_inf=ieee_features_type(feature=6_1)
type(ieee_features_type),parameter::ieee_invalid_flag=ieee_features_type(feature=7_1)
type(ieee_features_type),parameter::ieee_nan=ieee_features_type(feature=8_1)
type(ieee_features_type),parameter::ieee_rounding=ieee_features_type(feature=9_1)
type(ieee_features_type),parameter::ieee_sqrt=ieee_features_type(feature=10_1)
type(ieee_features_type),parameter::ieee_subnormal=ieee_features_type(feature=11_1)
type(ieee_features_type),parameter::ieee_underflow_flag=ieee_features_type(feature=12_1)
end
