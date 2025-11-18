import path from 'path';

const addonPath = path.resolve(__dirname, '..', '..', '..', 'build', 'Release', 'mlx.node');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const mlxModule = require(addonPath);

// Export the core object which contains all the functions
export default mlxModule.core;
