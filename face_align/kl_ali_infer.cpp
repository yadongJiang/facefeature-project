#include "kl_ali_infer.h"

#include "kl_alignface.h"

namespace klface {
    IKLAlignFace* IKLAlignFace::createKLAlignFace()
    {
 	return new KLAlignFace();
    }
}