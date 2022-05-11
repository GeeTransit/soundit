{#
Based on:
https://raw.githubusercontent.com/readthedocs/sphinx-autoapi/17ad4b988e6da41d3b9fc4fe253c19adc2fa3dc0/autoapi/templates/index.rst
#}


API Reference
=============

Auto-generated documentation from the code.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   {% for page in pages %}
   {% if page.top_level_object and page.display %}
   {{ page.include_path }}
   {% endif %}
   {% endfor %}
