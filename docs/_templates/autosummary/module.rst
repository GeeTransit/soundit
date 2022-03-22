{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
	:members:
	:undoc-members:
	:show-inheritance:
	:member-order: bysource

	{# The ordering of auto-summaries is from
	https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#confval-autoapi_member_order #}

	{% block modules %}
	{% if modules %}
	.. rubric:: {{ _("Modules") }}

	.. autosummary::
		:toctree:
		:recursive:
	{% for item in modules %}
		{{ item }}
	{%- endfor %}
	{% endif %}
	{% endblock %}

	{% block attributes %}
	{% if attributes %}
	.. rubric:: {{ _("Module Attributes") }}

	.. autosummary::
		:nosignatures:
	{% for item in attributes %}
		{{ item }}
	{%- endfor %}
	{% endif %}
	{% endblock %}

	{% block exceptions %}
	{% if exceptions %}
	.. rubric:: {{ _("Exceptions") }}

	.. autosummary::
		:nosignatures:
	{% for item in exceptions %}
		{{ item }}
	{%- endfor %}
	{% endif %}
	{% endblock %}

	{% block classes %}
	{% if classes %}
	.. rubric:: {{ _("Classes") }}

	.. autosummary::
		:nosignatures:
	{% for item in classes %}
		{{ item }}
	{%- endfor %}
	{% endif %}
	{% endblock %}

	{% block functions %}
	{% if functions %}
	.. rubric:: {{ _("Functions") }}

	.. autosummary::
		:nosignatures:
	{% for item in functions %}
		{{ item }}
	{%- endfor %}
	{% endif %}
	{% endblock %}

	{# This adds separation between the auto-summaries and the auto-docs. #}
	{% if members %}
	.. rubric:: {{ _("Documentation") }}
	{% endif %}
