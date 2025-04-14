# %% Import Modules and set constants
import sys

sys.path.append("../Functions")
import matplotlib.pyplot as plt
import plotting_func
import visibilities
from astropy import units as u
from astropy.coordinates import SkyCoord

# %%
phasecentre = SkyCoord(
    ra=56 * u.deg,
    dec=-30 * u.deg,
    equinox="J2000",
)
vt = visibilities.generate_visibilities(phasecentre=phasecentre)
# %%

u = vt.visibility_acc.u.data * 1.4e9 / 3e8
v = vt.visibility_acc.v.data * 1.4e9 / 3e8


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
plt.sca(ax)
ax.plot(u / 1000, v / 1000, ".", color="b", markersize=0.2, rasterized=True)
ax.plot(-u / 1000, -v / 1000, ".", color="b", markersize=0.2, rasterized=True)
ax.set(
    xlabel=r"$u$" + " [k" + r"$\lambda]$",
    ylabel=r"$v$" + " [k" + r"$\lambda]$",
    xlim=(-800, 800),
    ylim=(-800, 800),
)

ax.set_aspect("equal", adjustable="box")
ticks = [-750, -500, -250, 0, 250, 500, 750]
ax.set(
    xticks=ticks,
    yticks=ticks,
)

plotting_func.savefig("/home/tripathi/Main/Images/UV_coverage.pdf")
# plt.show()
