exec(open("squeeze.py").read().split("sim.start()")[0])
sim.start()
for k in range(30): sim.step()
print("per-contact stiffness array:", sim.contacts.rigid_contact_stiffness is not None)
mj = sim.solver.mjw_data; n = int(mj.nacon.numpy()[0])
geom = mj.contact.geom.numpy()[:n]; wid = mj.contact.worldid.numpy()[:n]
sr = mj.contact.solref.numpy()[:n]; si = mj.contact.solimp.numpy()[:n]; ds = mj.contact.dist.numpy()[:n]
g2s = sim.solver.mjc_geom_to_newton_shape.numpy()
spw = m.shape_count // nw
seen=set()
for i in range(n):
    w = wid[i]; key=(cases[w][0], cases[w][1])
    if cases[w][1] != 2 or key in seen: continue
    sh = [g2s[w, gg] if g2s.ndim == 2 else g2s[gg] for gg in geom[i]]
    seen.add(key)
    print(key, "shapes", sh, "solref", sr[i], "solimp", np.round(si[i], 5), "dist mm %.3f" % (ds[i]*1e3))
