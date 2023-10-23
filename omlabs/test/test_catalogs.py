import intake
import pkg_resources


def test_catalogs_are_present():
    f = pkg_resources.resource_filename("omlabs", "catalogs/obs_catalog_gfdl.yml")
    cat = intake.open_catalog(f)
    assert isinstance(cat, intake.catalog.local.YAMLFileCatalog)
