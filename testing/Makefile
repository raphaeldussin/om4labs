THETAO_BIAS = thetao_yz_atl_diff.png thetao_yz_gbl_diff.png thetao_yz_pac_diff.png

IMG = SST_diff.png SST_compare.png SSS_diff.png SSS_compare.png \
      moc.png seaice.nh.png heat_transport.png $(THETAO_BIAS)

all: $(IMG)

test_data:
	curl -o test_data.tar.gz ftp://ftp.gfdl.noaa.gov/pub/John.Krasting/om4labs/test_data.tar.gz
	tar -xzvf test_data.tar.gz
	rm -f test_data.tar.gz

heat_transport.png: test_data ../om4labs/diags/heat_transport/heat_transport.py
	om4labs heat_transport --model OM4p5 --platform testing -l OM4p5 \
	test_data/output/OM4p5/ocean_monthly.0001-0005.ann.nc

moc.png: test_data ../om4labs/diags/moc/moc.py
	om4labs moc --model OM4p125 --platform testing \
	test_data/output/ocean_month_z_d2_refined.0036-0040.ann.nc

SST_diff.png: test_data ../om4labs/diags/sst_annual_bias_1x1deg/sst_annual_bias_1x1deg.py
	om4labs sst_annual_bias_1x1deg -S diff --platform testing \
	test_data/output/ocean_annual_z_d2_1x1deg.0001-0010.ann.nc

SSS_diff.png: test_data ../om4labs/diags/sss_annual_bias_1x1deg/sss_annual_bias_1x1deg.py
	om4labs sss_annual_bias_1x1deg -S diff --platform testing \
	test_data/output/ocean_annual_z_d2_1x1deg.0001-0010.ann.nc

SST_compare.png: test_data ../om4labs/diags/sst_annual_bias_1x1deg/sst_annual_bias_1x1deg.py
	om4labs sst_annual_bias_1x1deg -S compare --platform testing \
	test_data/output/ocean_annual_z_d2_1x1deg.0001-0010.ann.nc

SSS_compare.png: test_data ../om4labs/diags/sss_annual_bias_1x1deg/sss_annual_bias_1x1deg.py
	om4labs sss_annual_bias_1x1deg -S compare --platform testing \
	test_data/output/ocean_annual_z_d2_1x1deg.0001-0010.ann.nc

seaice.nh.png: test_data ../om4labs/diags/seaice/seaice.py
	om4labs seaice --platform testing -m OM4p125 -O test_data/obs/OBS_NSIDC_sat_NH_T2Ms_sic.nc \
	test_data/output/ice_1x1deg.003601-004012.siconc_old.nc

$(THETAO_BIAS): test_data ../om4labs/diags/generic_yz_annual_bias_1x1deg/generic_yz_annual_bias_1x1deg.py
	om4labs thetao_yz_annual_bias_1x1deg --config OM4p125 --style diff --platform testing \
	test_data/output/ocean_annual_z_d2_1x1deg.0001-0010.ann.nc

clean:
	rm -f $(IMG)
	rm -f bilinear*.nc

distclean:
	rm -f $(IMG)
	rm -f bilinear*.nc
	rm -fR test_data
