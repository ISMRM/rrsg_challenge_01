data = h5read('cardiac_radial.h5','/data');
noise = h5read('cardiac_radial.h5','/noise');

data = data(1:2:end-1) + 1i*data(2:2:end);
data = reshape(data,[1 256 26 420]);
data = permute(data,[1 2 4 3]);

noise = noise(1:2:end-1) + 1i*noise(2:2:end);
noise = reshape(noise,[512 26 128 2]);
noise = permute(noise,[1 3 2]);

traj = bart('traj -r -G -x256 -y420');
traj_scaled = bart('scale 0.5', traj);
grid = bart('nufft -i -t', traj_rad2, data);
img = bart('rss 8',grid);
