{#
Based on:
https://raw.githubusercontent.com/readthedocs/sphinx-autoapi/17ad4b988e6da41d3b9fc4fe253c19adc2fa3dc0/autoapi/templates/python/data.rst
#}


{% if obj.display %}
.. py:{{ obj.type }}:: {{ obj.name }}
   {% if obj.annotation %}
   :type: {{ obj.annotation }}
   {% endif %}
   {% if obj.value is not none %}
   :value:
        {%- if obj.value is not none %} {%
            if obj.value is string and obj.value.splitlines()|count > 1 -%}
                Multiline-String

    .. raw:: html

        <details><summary>Show Value</summary>

    .. code-block:: text

        {{ obj.value|indent(width=8) }}

    .. raw:: html

        </details>

            {%- else -%}
                {{ "{!r}".format(obj.value)|truncate(100) }}
            {%- endif %}
        {%- endif %}
    {% endif %}


   {{ obj.docstring|indent(3) }}
{% endif %}
