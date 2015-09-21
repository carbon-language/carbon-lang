_CLC_OVERLOAD _CLC_DECL int get_image_width (image2d_t image);
_CLC_OVERLOAD _CLC_DECL int get_image_width (image3d_t image);

_CLC_OVERLOAD _CLC_DECL int get_image_height (image2d_t image);
_CLC_OVERLOAD _CLC_DECL int get_image_height (image3d_t image);

_CLC_OVERLOAD _CLC_DECL int get_image_depth (image3d_t image);

_CLC_OVERLOAD _CLC_DECL int get_image_channel_data_type (image2d_t image);
_CLC_OVERLOAD _CLC_DECL int get_image_channel_data_type (image3d_t image);

_CLC_OVERLOAD _CLC_DECL int get_image_channel_order (image2d_t image);
_CLC_OVERLOAD _CLC_DECL int get_image_channel_order (image3d_t image);

_CLC_OVERLOAD _CLC_DECL int2 get_image_dim (image2d_t image);
_CLC_OVERLOAD _CLC_DECL int4 get_image_dim (image3d_t image);
