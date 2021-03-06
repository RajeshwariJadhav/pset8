load ref4_scene4.mat;
size(reflectances)
slice = reflectances(:,:,17);
figure; imagesc(slice); colormap('gray'); brighten(0.5);
z = max(slice(100, 39));
slice_clip = min(slice, z)/z;
figure; imagesc(slice_clip.^0.4); colormap('gray');
reflectance = squeeze(reflectances(141, 75,:));
figure; plot(400:10:720, reflectance);
xlabel('wavelength, nm');
ylabel('unnormalized reflectance');
reflectances = reflectances/max(reflectances(:));
load illum_25000.mat;
radiances_25000 = zeros(size(reflectances)); % initialize array
for i = 1:33,
  radiances_25000(:,:,i) = reflectances(:,:,i)*illum_25000(i);
end
radiance_25000 = squeeze(radiances_25000(141, 75, :));
figure; plot(400:10:720, radiance_25000, 'b'); % blue curve
xlabel('wavelength, nm');
ylabel('radiance, arbitrary units');
hold on;
load illum_4000.mat;
radiances_4000 = zeros(size(reflectances)); % initialize array
for i = 1:33,
  radiances_4000(:,:,i) = reflectances(:,:,i)*illum_4000(i);
end
radiance_4000 = squeeze(radiances_4000(141, 75, :));
plot(400:10:720, radiance_4000, 'r'); % red curve

load illum_6500.mat;
radiances_6500 = zeros(size(reflectances)); % initialize array
for i = 1:33,
  radiances_6500(:,:,i) = reflectances(:,:,i)*illum_6500(i);
end
radiances = radiances_6500;
[r c w] = size(radiances);
radiances = reshape(radiances, r*c, w);
load xyzbar.mat;
XYZ = (xyzbar'*radiances')';
XYZ = reshape(XYZ, r, c, 3);
XYZ = max(XYZ, 0);
XYZ = XYZ/max(XYZ(:));
RGB = XYZ2sRGB_exgamma(XYZ);
RGB = max(RGB, 0);
RGB = min(RGB, 1);
figure; imshow(RGB.^0.4, 'Border','tight');

z = max(RGB(244,17,:));
RGB_clip = min(RGB, z)/z;
figure; imshow(RGB_clip.^0.4, 'Border','tight');
