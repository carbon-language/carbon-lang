#include <isl/printer.h>
#include <isl_yaml.h>

struct isl_printer_ops;

/* A printer to a file or a string.
 *
 * yaml_style is the YAML style in which the next elements should
 * be printed and may be either ISL_YAML_STYLE_BLOCK or ISL_YAML_STYLE_FLOW,
 * with ISL_YAML_STYLE_FLOW being the default.
 * yaml_state keeps track of the currently active YAML elements.
 * yaml_size is the size of this arrays, while yaml_depth
 * is the number of elements currently in use.
 * yaml_state may be NULL if no YAML printing is being performed.
 */
struct isl_printer {
	struct isl_ctx	*ctx;
	struct isl_printer_ops *ops;
	FILE        	*file;
	int		buf_n;
	int		buf_size;
	char		*buf;
	int		indent;
	int		output_format;
	char		*indent_prefix;
	char		*prefix;
	char		*suffix;
	int		width;

	int			yaml_style;
	int			yaml_depth;
	int			yaml_size;
	enum isl_yaml_state	*yaml_state;
};
