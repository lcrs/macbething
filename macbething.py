# To get the modules we need on CentOS 7:
# sudo yum install numpy openexr openexr-devel gcc-c++
# sudo easy_install -Z openexr (egg files end up in /usr/lib/python2.7/site-packages)
import OpenEXR, Imath, numpy, uuid

# Returns three numpy arrays from an EXR's RGB planes
def exr2arrays(exrpath):
	exr = OpenEXR.InputFile(exrpath)
	dw = exr.header()['dataWindow']
	w = dw.max.x - dw.min.x + 1
	h = dw.max.y - dw.min.y + 1
	ptfloat = Imath.PixelType(Imath.PixelType.FLOAT)
	r = numpy.fromstring(exr.channel('R', ptfloat), dtype=numpy.float32)
	r.shape = (h, w)
	g = numpy.fromstring(exr.channel('G', ptfloat), dtype=numpy.float32)
	g.shape = (h, w)
	b = numpy.fromstring(exr.channel('B', ptfloat), dtype=numpy.float32)
	b.shape = (h, w)
	return r, g, b

# Writes three numpy arrays to an EXR's RGB planes
def arrays2exr(r, g, b, exrpath):
	rstr = r.astype(numpy.float32).tostring()
	gstr = g.astype(numpy.float32).tostring()
	bstr = b.astype(numpy.float32).tostring()
	exr = OpenEXR.OutputFile(exrpath, OpenEXR.Header(r.shape[1], r.shape[0]))
	exr.writePixels({'R':rstr, 'G':gstr, 'B':bstr})
	exr.close()

# Returns a numpy array of integer patch IDs or 0 for no patch
# It's effectively an image telling us which pixels are in which chart patch
def patchids(w, h, samplesize, rows, cols, xsqueeze, ysqueeze):
	y, x = numpy.mgrid[0:h, 0:w]
	x = numpy.clip((1.0 + xsqueeze) * x - xsqueeze * (w/2.0), 0.0, w)
	y = numpy.clip((1.0 + ysqueeze) * y - ysqueeze * (h/2.0), 0.0, h)
	patchw, patchh = w/float(cols), h/float(rows)
	u, v = (x % patchw) / patchw, (y % patchh) / patchh
	t0, t1 = 0.5 - samplesize, 0.5 + samplesize
	patchmask = (u > t0) * (u < t1) * (v > t0) * (v < t1)
	patchx, patchy = numpy.floor(x/patchw), numpy.floor(y/patchh)
	ids = patchy * cols + patchx + 1
	ids *= patchmask
	ids = ids.astype(numpy.int32)
	return ids

# Returns three numpy arrays of patch RGB values, by averaging pixels under
# each integer patch ID in the ids array
def samplechart(r, g, b, ids):
	racc = numpy.bincount(ids.ravel(), weights=r.ravel())
	rcount = numpy.bincount(ids.ravel())
	ravg = racc / rcount
	gacc = numpy.bincount(ids.ravel(), weights=g.ravel())
	gcount = numpy.bincount(ids.ravel())
	gavg = gacc / gcount
	bacc = numpy.bincount(ids.ravel(), weights=b.ravel())
	bcount = numpy.bincount(ids.ravel())
	bavg = bacc / bcount
	return ravg, gavg, bavg

# Returns three numpy arrays forming RGB planes of a synthetic chart, shaped
# like the patch IDs in the ids array, with the patch values in sr, sg, sb
def makechart(sr, sg, sb, ids):
	r = sr[ids]
	g = sg[ids]
	b = sb[ids]
	return r, g, b

frontr, frontg, frontb = exr2arrays('front.exr')
targetr, targetg, targetb = exr2arrays('target.exr')
ids = patchids(frontr.shape[1], frontr.shape[0], 0.4, 4, 6, 0.025, 0.095)
fr, fg, fb = samplechart(frontr, frontg, frontb, ids)
tr, tg, tb = samplechart(targetr, targetg, targetb, ids)
frontsamples = zip(fr, fg, fb)
targetsamples = zip(tr, tg, tb)

frontvalidpatches = []
targetvalidpatches = []
rejects = 0
for i in range(1, len(frontsamples)):
	if(i == 0):
		continue # ID 0 is all pixels not in a patch
	if(min(frontsamples[i] + targetsamples[i]) <= 0.0):
		rejects = rejects + 1
		continue # This patch is clipped
	frontvalidpatches.append(frontsamples[i])
	targetvalidpatches.append(targetsamples[i])

mat = numpy.linalg.lstsq(frontvalidpatches, targetvalidpatches)[0].transpose()
nukecolormatrix = 'ColorMatrix {\n matrix { {%f %f %f} {%f %f %f} {%f %f %f} }\n label "Created from\\ncolour chart\\nby Ls_LUTy"\n}\n' % tuple(mat.ravel())
ctf = '<?xml version="1.0" encoding="UTF-8"?>\n<ProcessList id="%s" version="1.2">\n    <Description>Matrix created from colour chart by Ls_LUTy</Description>\n    <Matrix inBitDepth="16f" outBitDepth="16f">\n        <Array dim="3 3 3">\n %f %f %f\n %f %f %f\n %f %f %f\n        </Array>\n    </Matrix>\n</ProcessList>\n' % ((str(uuid.uuid4()),) + tuple(mat.ravel()))
