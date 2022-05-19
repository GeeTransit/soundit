.. program-include:: gitchangelog
    :cwd: /..
    :env: GITCHANGELOG_CONFIG_FILENAME=docs/gitchangelog_conf.py

.. rebuild-when:: file-change

    /gitchangelog_conf.py

.. rebuild-when:: program-change

    hatch version
