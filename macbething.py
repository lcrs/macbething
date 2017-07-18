# To get the modules we need:
# sudo yum install numpy openexr openexr-devel gcc-c++
# sudo easy_install -Z openexr
#   (openexr egg ends up in /usr/lib/python2.7/site-packages)

import OpenEXR, Imath, numpy
samplesize = 0.43

frontexr = OpenEXR.InputFile('t.exr')
dw = frontexr.header()['dataWindow']
w = dw.max.x - dw.min.x + 1
h = dw.max.y - dw.min.y + 1
ptfloat = Imath.PixelType(Imath.PixelType.FLOAT)

frontr = numpy.fromstring(frontexr.channel('R', ptfloat), dtype=numpy.float32)
frontr.shape = (h, w)
frontg = numpy.fromstring(frontexr.channel('G', ptfloat), dtype=numpy.float32)
frontg.shape = (h, w)
frontb = numpy.fromstring(frontexr.channel('B', ptfloat), dtype=numpy.float32)
frontb.shape = (h, w)

y, x = numpy.mgrid[:h, :w]
x = numpy.clip(1.025 * x - 0.025 * (w/2.0), 0.0, w)
y = numpy.clip(1.090 * y - 0.090 * (h/2.0), 0.0, h)
patchw, patchh = w/6.0, h/4.0
u, v = (x % patchw) / patchw, (y % patchh) / patchh
t0, t1 = 0.5 - samplesize, 0.5 + samplesize
patchmask = (u > t0) * (u < t1) * (v > t0) * (v < t1)
patchx, patchy = numpy.floor(x/patchw), numpy.floor(y/patchh)
patchids = patchy * 6 + patchx + 1
patchids *= patchmask
patchids = patchids.astype(numpy.int32)

sampleaccums = numpy.bincount(patchids.ravel(), weights=frontr.ravel())
samplecounts = numpy.bincount(patchids.ravel())
sampleavgs = sampleaccums / samplecounts
ravg = sampleavgs[patchids]
sampleaccums = numpy.bincount(patchids.ravel(), weights=frontg.ravel())
samplecounts = numpy.bincount(patchids.ravel())
sampleavgs = sampleaccums / samplecounts
gavg = sampleavgs[patchids]
sampleaccums = numpy.bincount(patchids.ravel(), weights=frontb.ravel())
samplecounts = numpy.bincount(patchids.ravel())
sampleavgs = sampleaccums / samplecounts
bavg = sampleavgs[patchids]

resultrstr = ravg.astype(numpy.float32).tostring()
resultgstr = gavg.astype(numpy.float32).tostring()
resultbstr = bavg.astype(numpy.float32).tostring()
resultexr = OpenEXR.OutputFile('u.exr', OpenEXR.Header(w, h))
resultexr.writePixels({'R':resultrstr, 'G':resultgstr, 'B':resultbstr})
resultexr.close()
