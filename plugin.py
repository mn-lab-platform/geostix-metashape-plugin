# This install should be executed only once with first run.
from modules.pip_auto_install import pip_install

pip_install("numpy\npandas\nscipy\npillow\nmatplotlib", ask=False)

import Metashape

import plugin_align_and_analyse
import plugin_import_geostix

if __name__ == "__main__":
    application: Metashape.Application = Metashape.app

    application.removeMenuItem(plugin_import_geostix.MENU_ITEM_NAME)
    application.addMenuItem(plugin_import_geostix.MENU_ITEM_NAME, plugin_import_geostix.import_geostix_tool)

    application.removeMenuItem(plugin_align_and_analyse.MENU_ITEM_NAME)
    application.addMenuItem(plugin_align_and_analyse.MENU_ITEM_NAME, plugin_align_and_analyse.align_and_analyse)
